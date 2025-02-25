import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from collections import OrderedDict
from imblearn.over_sampling import SMOTE
import optuna
import warnings
import random
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
warnings.filterwarnings('ignore')

# 1. 설정
SEED = 42
N_SPLITS = 10       # Fold 개수
N_TRIALS = 100     # Fold당 Optuna 최적화 반복 수
BATCH_SIZE = 2048
EPOCHS = 50
PATIENCE = 8
NUM_WORKERS = 4

# 앙상블 가중치 초기화 -> 추후에 그리드 서치로 바뀜
CATBOOST_WEIGHT = 0.5
LGBM_WEIGHT = 0.3
TABNET_WEIGHT = 0.2

# GPU 설정
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 재현성을 위한 시드 설정
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(SEED)

# 2. 데이터 로드
print("데이터 로드 중...")
train_path = "../data/processed_train_ver6.csv"
test_path = "../data/processed_test_ver6.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X = train_df.drop("임신 성공 여부", axis=1)
y = train_df["임신 성공 여부"]
X_test = test_df.copy()

test_ids = X_test["ID"].copy()
X_test = X_test.drop(columns=["ID"])

# 3. 범주형 컬럼 식별
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# 정수형 범주형 변수들 추가
categorical_cols += ["총 시술 횟수_구간", "클리닉 내 총 시술 횟수_구간", "배란 자극 여부", "단일 배아 이식 여부",
                     "착상 전 유전 진단 사용 여부", "동결 배아 사용 여부", "신선 배아 사용 여부", "기증 배아 사용 여부",
                     "대리모 여부", "불명확 불임 원인", "불임 원인 - 난관 질환", "불임 원인 - 남성 요인", 
                     "불임 원인 - 배란 장애", "불임 원인 - 자궁내막증",
                     "시술_과정_완성도", "통합_주_불임", "통합_부_불임", "통합_정자_문제",
                     "배아 해동 경과일", "난자 채취 경과일", "난자 혼합 경과일", "미세주입 후 저장된 배아 수_cat", 
                     "해동된 배아 수_cat", "해동 난자 수_cat", "저장된 신선 난자 수_cat", "기증자 정자와 혼합된 난자 수_cat",
                     "저장된 배아 수", "수집된 신선 난자 수", "혼합된 난자 수", "파트너 정자와 혼합된 난자 수",
                     "총 생성 배아 수", "미세주입된 난자 수", "미세주입에서 생성된 배아 수", "이식된 배아 수", 
                     "미세주입 배아 이식 수"]

# 4. 범주형 컬럼을 category 타입으로 변환
for col in categorical_cols:
    if col in X.columns and col in X_test.columns:
        X[col] = X[col].astype('category')
        X_test[col] = X_test[col].astype('category')

# 5. TabNet 모델 관련 클래스 정의
# 5.1 Ghost Batch Normalization
class GhostBatchNorm(nn.Module):
    def __init__(self, num_features, virtual_batch_size=128, momentum=0.1):
        super(GhostBatchNorm, self).__init__()
        self.num_features = num_features
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(num_features, momentum=momentum)
        
    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = []
        for chunk in chunks:
            res.append(self.bn(chunk))
        return torch.cat(res, dim=0)

# 5.2 GLU
class GLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GLU, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
        
    def forward(self, x):
        x = self.fc(x)
        x, gate = x.chunk(2, dim=1)
        return x * torch.sigmoid(gate)

# 5.3 AttentionTransformer
class AttentionTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, virtual_batch_size=128, momentum=0.1):
        super(AttentionTransformer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = GhostBatchNorm(output_dim, virtual_batch_size, momentum)
        
    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        return torch.sigmoid(x)

# 5.4 FeatureTransformer
class FeatureTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, virtual_batch_size=128, momentum=0.1):
        super(FeatureTransformer, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.bn1 = GhostBatchNorm(output_dim, virtual_batch_size, momentum)
        self.glu = GLU(output_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.glu(x)
        return x

# 5.5 TabNetEncoder
class TabNetEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, feature_dims=64, attention_dims=64, 
                 num_steps=3, virtual_batch_size=128, momentum=0.1):
        super(TabNetEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feature_dims = feature_dims
        self.attention_dims = attention_dims
        self.num_steps = num_steps
        
        self.initial_fc = nn.Linear(input_dim, feature_dims)
        self.initial_bn = GhostBatchNorm(feature_dims, virtual_batch_size, momentum)
        
        self.attentions = nn.ModuleList([
            AttentionTransformer(
                feature_dims, feature_dims, virtual_batch_size, momentum
            ) for _ in range(num_steps)
        ])
        
        self.feature_transformers = nn.ModuleList([
            FeatureTransformer(
                feature_dims, feature_dims, virtual_batch_size, momentum
            ) for _ in range(num_steps)
        ])
        
        self.final_fc = nn.Linear(feature_dims, output_dim)
        
    def forward(self, x):
        # Initial processing
        x = self.initial_fc(x)
        x = self.initial_bn(x)
        
        prior_scales = torch.ones(x.shape).to(x.device)
        M_loss = 0
        
        attentions = []
        features = []
        
        for step in range(self.num_steps):
            # Apply attention
            attention = self.attentions[step](x)
            attentions.append(attention)
            
            # Apply prior scale
            attention = attention * prior_scales
            prior_scales = prior_scales * (1 - attention)
            
            # Apply feature transformer 
            masked_x = x * attention
            features.append(masked_x)
            x_out = self.feature_transformers[step](masked_x)
            
            # Update feature representation
            x = x + x_out
            
            # Calculate sparsity loss
            M_loss += torch.mean(torch.sum(attention, dim=1))
        
        # Final processing
        x = self.final_fc(x)
        return x, M_loss, attentions, features

# 5.6 전처리 클래스
class PreprocessPipeline:
    def __init__(self, categorical_cols=None):
        self.categorical_cols = [] if categorical_cols is None else categorical_cols
        self.numerical_cols = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categorical_dims = {}
        self.embedding_dims = {}
    
    def fit(self, X):
        # 수치형 컬럼 식별
        self.numerical_cols = [col for col in X.columns if col not in self.categorical_cols]
        
        # 수치형 특성 스케일링을 위한 scaler 학습
        if self.numerical_cols:
            numerical_data = X[self.numerical_cols].fillna(X[self.numerical_cols].median())
            self.scaler.fit(numerical_data)
        
        # 범주형 특성을 위한 label encoder 학습
        for col in self.categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                # 데이터 복사 및 문자열로 변환 (category 타입 보존을 위해)
                X_col = X[col].copy().astype(str).fillna('Unknown')
                # 레이블 인코더 학습
                le.fit(X_col)
                self.label_encoders[col] = le
                # 임베딩 차원 계산
                cardinality = len(le.classes_)
                self.categorical_dims[col] = cardinality
                self.embedding_dims[col] = min(max(2, int((cardinality)**0.25)), 50)
        
        return self
    
    def transform(self, X):
        # 원본 데이터는 변경하지 않기 위해 복사
        X_copy = X.copy()
        
        # 범주형 특성 변환
        cat_features = []
        for col in self.categorical_cols:
            if col in X_copy.columns and col in self.label_encoders:
                # 복사한 데이터를 문자열로 변환하고 NA 값을 'Unknown'으로 대체
                X_col = X_copy[col].astype(str).fillna('Unknown')
                
                # 훈련 데이터에 없던 새로운 범주는 'Unknown'으로 대체
                unknown_values = ~X_col.isin(self.label_encoders[col].classes_)
                if unknown_values.any():
                    X_col.loc[unknown_values] = 'Unknown'
                
                # 인코딩 적용
                try:
                    encoded_values = self.label_encoders[col].transform(X_col)
                    cat_features.append(encoded_values)
                except ValueError as e:
                    # 예외 처리 - 알 수 없는 값 대체
                    print(f"Warning: {e} in column {col}")
                    # 기본값(첫 번째 클래스)으로 대체
                    default_values = np.zeros(len(X_col))
                    cat_features.append(default_values)
        
        # 수치형 특성 변환
        num_features = []
        if self.numerical_cols:
            numerical_data = X_copy[self.numerical_cols].copy()
            # 결측치 처리
            for col in self.numerical_cols:
                numerical_data[col] = numerical_data[col].fillna(numerical_data[col].median())
            # 스케일링 적용
            num_data = self.scaler.transform(numerical_data)
            num_features.append(num_data)
        
        # 변환된 특성 반환 (원본 데이터는 변경되지 않음)
        return X_copy, np.column_stack(cat_features) if cat_features else np.array([]), np.column_stack(num_features) if num_features else np.array([])

# 5.7 TabNet 모델
class TabNetModel(nn.Module):
    def __init__(self, categorical_dims, embedding_dims, numerical_dims, output_dim=1):
        super(TabNetModel, self).__init__()
        
        # 임베딩 레이어
        self.embeddings = nn.ModuleList([
            nn.Embedding(categorical_dims[col], embedding_dims[col]) 
            for col in categorical_dims
        ])
        
        # 임베딩 드롭아웃
        self.emb_dropout = nn.Dropout(0.2)
        
        # 임베딩 차원의 합 계산
        total_emb_dims = sum(embedding_dims.values())
        
        # TabNet 레이어
        combined_dims = total_emb_dims + numerical_dims
        hidden_dims = 128
        
        self.tabnet = TabNetEncoder(
            input_dim=combined_dims,
            output_dim=hidden_dims,
            feature_dims=hidden_dims,
            attention_dims=hidden_dims,
            num_steps=5,
            virtual_batch_size=256
        )
        
        # Final layers
        self.final_bn = nn.BatchNorm1d(hidden_dims)
        self.final_dropout = nn.Dropout(0.3)
        self.final_fc = nn.Linear(hidden_dims, output_dim)
        
    def forward(self, categorical_x, numerical_x):
        # 범주형 특성 임베딩
        if categorical_x.size(1) > 0:
            embeddings = [emb(categorical_x[:, i]) for i, emb in enumerate(self.embeddings)]
            x = torch.cat(embeddings, 1)
            x = self.emb_dropout(x)
        else:
            x = torch.tensor([]).to(numerical_x.device)
            
        # 수치형 특성과 결합
        if numerical_x.size(1) > 0:
            if x.size(0) > 0:
                x = torch.cat([x, numerical_x], 1)
            else:
                x = numerical_x
        
        # TabNet 인코더 적용
        x, M_loss, _, _ = self.tabnet(x)
        
        # 최종 분류층
        x = self.final_bn(x)
        x = F.relu(x)
        x = self.final_dropout(x)
        x = self.final_fc(x)
        
        return torch.sigmoid(x).squeeze(1), M_loss

# 5.8 Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# 5.9 데이터셋 클래스
class TabularDataset(Dataset):
    def __init__(self, cat_features, num_features, labels=None):
        self.cat_features = torch.tensor(cat_features, dtype=torch.long) if cat_features.size > 0 else torch.zeros((cat_features.shape[0], 0), dtype=torch.long)
        self.num_features = torch.tensor(num_features, dtype=torch.float32) if num_features.size > 0 else torch.zeros((num_features.shape[0], 0), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None
    
    def __len__(self):
        return len(self.cat_features)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.cat_features[idx], self.num_features[idx], self.labels[idx]
        else:
            return self.cat_features[idx], self.num_features[idx]

# 6. 모델 훈련 및 예측 함수
# 6.1 TabNet 모델 훈련 함수
def train_tabnet_model(model, train_loader, val_loader, optimizer, criterion, scheduler, device, 
                model_path, epochs=EPOCHS, patience=PATIENCE):
    # Early stopping 설정
    best_val_auc = 0
    patience_counter = 0
    best_epoch = 0
    
    # 훈련 및 검증 손실/메트릭 기록
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_auc': [],
        'val_auc': [],
        'sparsity_loss': []
    }
    
    for epoch in range(epochs):
        # 훈련 모드
        model.train()
        train_loss = 0
        sparsity_loss = 0
        train_preds = []
        train_targets = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for cat_data, num_data, targets in pbar:
            cat_data = cat_data.to(device)
            num_data = num_data.to(device)
            targets = targets.to(device)
            
            # 그래디언트 초기화
            optimizer.zero_grad()
            
            # 순방향 전파
            outputs, m_loss = model(cat_data, num_data)
            
            # 손실 계산
            loss = criterion(outputs, targets) + 0.0001 * m_loss
            
            # 역방향 전파
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 가중치 업데이트
            optimizer.step()
            
            # 훈련 손실 누적
            train_loss += loss.item() * targets.size(0)
            sparsity_loss += m_loss.item()
            
            # 예측 및 타겟 저장
            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
            
            # 진행 상황 업데이트
            pbar.set_postfix({'loss': loss.item()})
        
        # 평균 훈련 손실 및 AUC 계산
        train_loss /= len(train_loader.dataset)
        train_auc = roc_auc_score(train_targets, train_preds)
        sparsity_loss /= len(train_loader)
        
        # 검증 모드
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for cat_data, num_data, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]"):
                cat_data = cat_data.to(device)
                num_data = num_data.to(device)
                targets = targets.to(device)
                
                # 예측
                outputs, _ = model(cat_data, num_data)
                
                # 손실 계산
                loss = criterion(outputs, targets)
                
                # 검증 손실 누적
                val_loss += loss.item() * targets.size(0)
                
                # 예측 및 타겟 저장
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        # 평균 검증 손실 및 AUC 계산
        val_loss /= len(val_loader.dataset)
        val_auc = roc_auc_score(val_targets, val_preds)
        
        # 학습률 스케줄러 업데이트
        scheduler.step(val_loss)
        
        # 결과 출력
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {train_loss:.6f} - Train AUC: {train_auc:.6f} - '
              f'Val Loss: {val_loss:.6f} - Val AUC: {val_auc:.6f} - '
              f'Sparsity Loss: {sparsity_loss:.6f}')
        
        # 메트릭 기록
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        history['sparsity_loss'].append(sparsity_loss)
        
        # 최고 AUC 모델 저장
        if val_auc > best_val_auc:
            print(f"Validation AUC improved from {best_val_auc:.6f} to {val_auc:.6f}")
            best_val_auc = val_auc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} as validation AUC didn't improve for {patience} epochs")
                break
    
    print(f"Best model at epoch {best_epoch} with validation AUC: {best_val_auc:.6f}")
    
    # 최적의 모델 로드
    model.load_state_dict(torch.load(model_path))
    
    return model, history, best_val_auc

# 6.2 TabNet 예측 함수
def predict_with_tabnet(model, dataloader, device):
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for cat_data, num_data in tqdm(dataloader, desc="Predicting with TabNet"):
            cat_data = cat_data.to(device)
            num_data = num_data.to(device)
            outputs, _ = model(cat_data, num_data)
            all_preds.extend(outputs.cpu().numpy())
    
    return np.array(all_preds)

# 6.3 LightGBM Objective 함수
def objective_lgb(trial, X_train, y_train, X_val, y_val, class_weight_dict):
    params = {
        'objective': 'binary',
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': SEED,
        'class_weight': class_weight_dict,
        'verbose': -1
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='auc',
              callbacks=[lgb.early_stopping(50)])
    preds = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)

# 6.4 CatBoost Objective 함수
def objective_cat(trial, X_train, y_train, X_val, y_val, categorical_cols, class_weight_dict):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_seed': SEED,
        'class_weights': class_weight_dict,
        'verbose': 0
    }
    model = CatBoostClassifier(**params, 
                              cat_features=categorical_cols, 
                              eval_metric='AUC', 
                              task_type="GPU")
    model.fit(X_train, y_train,
             eval_set=(X_val, y_val),
             early_stopping_rounds=50,
             verbose=0)
    preds = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)

# 6.5 앙상블 그리드 서치 함수
def find_optimal_weights(cat_preds, lgb_preds, tabnet_preds, true_labels):
    best_auc = 0
    best_weights = (1/3, 1/3, 1/3)  # 기본 동일 가중치
    
    # 가중치 그리드 정의 (합계가 1이 되게 설정)
    weight_grid = []
    step = 0.05
    for cat_w in np.arange(0.1, 0.6 + step, step):
        for lgb_w in np.arange(0.1, 0.6 + step, step):
            tabnet_w = 1 - cat_w - lgb_w
            if 0.1 <= tabnet_w <= 0.6:
                weight_grid.append((cat_w, lgb_w, tabnet_w))
    
    print(f"총 {len(weight_grid)}개 가중치 조합 테스트 중...")
    
    # 각 가중치 조합 테스트
    for cat_w, lgb_w, tabnet_w in weight_grid:
        ensemble_preds = (
            cat_preds * cat_w +
            lgb_preds * lgb_w +
            tabnet_preds * tabnet_w
        )
        auc_score = roc_auc_score(true_labels, ensemble_preds)
        
        if auc_score > best_auc:
            best_auc = auc_score
            best_weights = (cat_w, lgb_w, tabnet_w)
    
    print(f"최적 가중치 발견: CatBoost={best_weights[0]:.2f}, LightGBM={best_weights[1]:.2f}, TabNet={best_weights[2]:.2f}")
    print(f"최적 앙상블 AUC: {best_auc:.6f}")
    return best_weights, best_auc

# 7. 하이브리드 앙상블 메인 함수
def main():
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    results = []
    fold_predictions = {}
    
    lgb_test_preds_total = np.zeros(len(X_test))
    cat_test_preds_total = np.zeros(len(X_test))
    tabnet_test_preds_total = np.zeros(len(X_test))
    ensemble_test_preds_total = np.zeros(len(X_test))
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'='*20} Fold {fold}/{N_SPLITS} {'='*20}")
        
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train_fold), y=y_train_fold)
        class_weight_dict = {i: w for i, w in zip(np.unique(y_train_fold), class_weights)}
        
        # 1. CatBoost 모델 학습
        print("1. CatBoost 하이퍼파라미터 최적화 중...")
        study_cat = optuna.create_study(direction='maximize')
        study_cat.optimize(lambda trial: objective_cat(trial, X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                                                      categorical_cols, class_weight_dict), n_trials=N_TRIALS)
        best_cat_params = study_cat.best_trial.params
        print(f"Best CatBoost parameters: {best_cat_params}")
        
        print("CatBoost 모델 학습 중...")
        final_cat_model = CatBoostClassifier(**best_cat_params, cat_features=categorical_cols, random_seed=SEED, eval_metric='AUC', task_type="GPU", class_weights=class_weight_dict)
        final_cat_model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold), early_stopping_rounds=50, verbose=100)
        
        # 2. LightGBM 모델 학습
        print("2. LightGBM 하이퍼파라미터 최적화 중...")
        study_lgb = optuna.create_study(direction='maximize')
        study_lgb.optimize(lambda trial: objective_lgb(trial, X_train_fold, y_train_fold, X_val_fold, y_val_fold, class_weight_dict), n_trials=N_TRIALS)
        best_lgb_params = study_lgb.best_trial.params
        print(f"Best LightGBM parameters: {best_lgb_params}")
        
        print("LightGBM 모델 학습 중...")
        final_lgb_model = lgb.LGBMClassifier(**best_lgb_params, random_state=SEED, class_weight=class_weight_dict)
        final_lgb_model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], eval_metric='auc', callbacks=[lgb.early_stopping(50)])
        
        # 3. TabNet 모델 전처리 및 학습
        print("3. TabNet 모델 전처리 중...")
        preprocess = PreprocessPipeline(categorical_cols=categorical_cols)
        preprocess.fit(X_train_fold)
        _, X_train_cat, X_train_num = preprocess.transform(X_train_fold)
        _, X_val_cat, X_val_num = preprocess.transform(X_val_fold)
        _, X_test_cat, X_test_num = preprocess.transform(X_test)
        
        train_dataset = TabularDataset(X_train_cat, X_train_num, y_train_fold.values)
        val_dataset = TabularDataset(X_val_cat, X_val_num, y_val_fold.values)
        test_dataset = TabularDataset(X_test_cat, X_test_num)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        
        tabnet_model = TabNetModel(categorical_dims=preprocess.categorical_dims, embedding_dims=preprocess.embedding_dims, numerical_dims=X_train_num.shape[1]).to(DEVICE)
        
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        optimizer = optim.AdamW(tabnet_model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        print("TabNet 모델 학습 중...")
        model_path = f'models/tabnet_fold{fold}.pth'
        tabnet_model, history, tabnet_val_auc = train_tabnet_model(tabnet_model, train_loader, val_loader, optimizer, criterion, scheduler, DEVICE, model_path, epochs=EPOCHS, patience=PATIENCE)
        
        # 4. 검증 세트에 대한 예측
        cat_val_preds = final_cat_model.predict_proba(X_val_fold)[:, 1]
        lgb_val_preds = final_lgb_model.predict_proba(X_val_fold)[:, 1]
        tabnet_val_preds = []
        tabnet_model.eval()
        with torch.no_grad():
            for cat_data, num_data, _ in val_loader:
                cat_data, num_data = cat_data.to(DEVICE), num_data.to(DEVICE)
                outputs, _ = tabnet_model(cat_data, num_data)
                tabnet_val_preds.extend(outputs.cpu().numpy())
        tabnet_val_preds = np.array(tabnet_val_preds)
        
        # 5. 최적 앙상블 가중치 찾기 (검증 데이터 기반)
        print("최적 앙상블 가중치 검색 중...")
        (cat_weight, lgb_weight, tabnet_weight), _ = find_optimal_weights(cat_val_preds, lgb_val_preds, tabnet_val_preds, y_val_fold)
        
        # 6. 최적 가중치로 검증 앙상블 예측
        ensemble_val_preds = cat_val_preds * cat_weight + lgb_val_preds * lgb_weight + tabnet_val_preds * tabnet_weight
        ensemble_val_auc = roc_auc_score(y_val_fold, ensemble_val_preds)
        
        # 7. 테스트 세트에 대한 예측 (각 모델)
        cat_test_preds = final_cat_model.predict_proba(X_test)[:, 1]
        lgb_test_preds = final_lgb_model.predict_proba(X_test)[:, 1]
        tabnet_test_preds = predict_with_tabnet(tabnet_model, test_loader, DEVICE)
        
        # 8. 최적 가중치로 테스트 앙상블 예측
        ensemble_test_preds = cat_test_preds * cat_weight + lgb_test_preds * lgb_weight + tabnet_test_preds * tabnet_weight
        
        # 폴드 결과 저장
        fold_result = {}
        fold_result["Fold"] = fold
        fold_result["CatBoost_AUC"] = roc_auc_score(y_val_fold, cat_val_preds)
        fold_result["LightGBM_AUC"] = roc_auc_score(y_val_fold, lgb_val_preds)
        fold_result["TabNet_AUC"] = tabnet_val_auc
        fold_result["Ensemble_AUC"] = ensemble_val_auc
        fold_result["CatBoost_Weight"] = cat_weight
        fold_result["LightGBM_Weight"] = lgb_weight
        fold_result["TabNet_Weight"] = tabnet_weight
        
        print(f"Fold {fold} 검증 AUC:")
        print(f"- CatBoost: {fold_result['CatBoost_AUC']:.6f}")
        print(f"- LightGBM: {fold_result['LightGBM_AUC']:.6f}")
        print(f"- TabNet: {fold_result['TabNet_AUC']:.6f}")
        print(f"- Ensemble: {fold_result['Ensemble_AUC']:.6f}")
        
        fold_predictions[fold] = {
            'catboost': cat_test_preds,
            'lightgbm': lgb_test_preds,
            'tabnet': tabnet_test_preds,
            'ensemble': ensemble_test_preds,
            'catboost_val_auc': fold_result["CatBoost_AUC"],
            'lightgbm_val_auc': fold_result["LightGBM_AUC"],
            'tabnet_val_auc': fold_result["TabNet_AUC"],
            'ensemble_val_auc': fold_result["Ensemble_AUC"]
        }
        
        results.append(fold_result)
        
        lgb_test_preds_total += lgb_test_preds / N_SPLITS
        cat_test_preds_total += cat_test_preds / N_SPLITS
        tabnet_test_preds_total += tabnet_test_preds / N_SPLITS
        ensemble_test_preds_total += ensemble_test_preds / N_SPLITS
        
        fold_submission = pd.DataFrame({"ID": test_ids, "probability": ensemble_test_preds})
        fold_submission.to_csv(f"results/hybrid_ensemble_fold{fold}_{fold_result['Ensemble_AUC']:.6f}.csv", index=False)
        print(f"- 예측 결과가 results/hybrid_ensemble_fold{fold}_{fold_result['Ensemble_AUC']:.6f}.csv 로 저장되었습니다.")
        
        # 9. 모델 시각화
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(f'Fold {fold} Loss Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(history['train_auc'], label='Train AUC')
        plt.plot(history['val_auc'], label='Validation AUC')
        plt.title(f'Fold {fold} AUC Curves')
        plt.xlabel('Epochs')
        plt.ylabel('AUC')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        model_names = ['CatBoost', 'LightGBM', 'TabNet', 'Ensemble']
        model_aucs = [fold_result['CatBoost_AUC'], fold_result['LightGBM_AUC'], fold_result['TabNet_AUC'], fold_result['Ensemble_AUC']]
        plt.bar(model_names, model_aucs)
        plt.title(f'Fold {fold} Model Comparison')
        plt.ylabel('Validation AUC')
        plt.ylim(0.5, 1.0)
        for i, v in enumerate(model_aucs):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(f'results/fold{fold}_comparison.png')
        plt.close()
    
    # 최종 테스트 결과 저장
    cat_submission = pd.DataFrame({"ID": test_ids, "probability": cat_test_preds_total})
    cat_submission.to_csv("results/catboost_final.csv", index=False)
    
    lgb_submission = pd.DataFrame({"ID": test_ids, "probability": lgb_test_preds_total})
    lgb_submission.to_csv("results/lightgbm_final.csv", index=False)
    
    tabnet_submission = pd.DataFrame({"ID": test_ids, "probability": tabnet_test_preds_total})
    tabnet_submission.to_csv("results/tabnet_final.csv", index=False)
    
    ensemble_submission = pd.DataFrame({"ID": test_ids, "probability": ensemble_test_preds_total})
    ensemble_submission.to_csv("results/hybrid_ensemble_final.csv", index=False)
    
    print("\n최종 예측 결과가 저장되었습니다:")
    print("- CatBoost: results/catboost_final.csv")
    print("- LightGBM: results/lightgbm_final.csv")
    print("- TabNet: results/tabnet_final.csv")
    print("- Hybrid Ensemble: results/hybrid_ensemble_final.csv")
    
    df_results = pd.DataFrame(results)
    df_results.to_csv("results/hybrid_ensemble_fold_results.csv", index=False)
    print("\nFold별 성능 기록이 results/hybrid_ensemble_fold_results.csv 로 저장되었습니다.")
    
    print("\n전체 Fold 성능 요약:")
    print(f"평균 CatBoost AUC: {df_results['CatBoost_AUC'].mean():.6f}")
    print(f"평균 LightGBM AUC: {df_results['LightGBM_AUC'].mean():.6f}")
    print(f"평균 TabNet AUC: {df_results['TabNet_AUC'].mean():.6f}")
    print(f"평균 Hybrid Ensemble AUC: {df_results['Ensemble_AUC'].mean():.6f}")
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='variable', y='value', data=pd.melt(df_results, id_vars=['Fold'], value_vars=['CatBoost_AUC', 'LightGBM_AUC', 'TabNet_AUC', 'Ensemble_AUC'], var_name='variable', value_name='value'))
    plt.title('Model Performance Comparison Across Folds')
    plt.xlabel('Model')
    plt.ylabel('AUC')
    plt.ylim(0.5, 1.0)
    plt.tight_layout()
    plt.savefig('results/overall_model_comparison.png')
    plt.close()
    
    plt.figure(figsize=(8, 5))
    plt.pie([cat_weight, lgb_weight, tabnet_weight], labels=['CatBoost', 'LightGBM', 'TabNet'], autopct='%1.1f%%', startangle=90, explode=(0.1, 0, 0))
    plt.title('Ensemble Model Weights')
    plt.tight_layout()
    plt.savefig('results/ensemble_weights.png')
    
    print("\n모델 비교 시각화가 저장되었습니다:")
    print("- results/overall_model_comparison.png")
    print("- results/ensemble_weights.png")

if __name__ == "__main__":
    main()
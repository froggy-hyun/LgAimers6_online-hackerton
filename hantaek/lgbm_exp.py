import pandas as pd
import numpy as np
import os
import optuna
import joblib
import time
import lightgbm as lgb
import random
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import warnings

warnings.filterwarnings("ignore")

# SEED 설정
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_DIR = "./weight/lightgbm"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    """ 학습 데이터를 로드하고 feature(X), label(y) 분리 """
    df = pd.read_csv("../data/processed_train_ver2.csv")
    X = df.drop('임신 성공 여부', axis=1)
    y = df['임신 성공 여부']

    # 클래스 가중치 계산
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
    class_weights_dict = {int(cls): weight for cls, weight in zip(np.unique(y), class_weights)}
    print(f"🔹 클래스 가중치 적용: {class_weights_dict}")
    return X, y, class_weights_dict


def objective_lgbm(trial, X, y, class_weights, n_splits=10):
    """ Optuna를 활용한 LightGBM 하이퍼파라미터 탐색 """
    params = {
        "objective": "binary",
        "metric": "auc",
        "random_state": SEED,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50, step=5),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.1)
    }
    
    # 10-Fold 교차 검증 설정
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    auc_scores = []
    
    for train_idx, valid_idx in kf.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        cw = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: w for i, w in enumerate(cw)}
        
        model = lgb.LGBMClassifier(**params, class_weight=class_weight_dict)
        model.fit(
            X_train, 
            y_train, 
            eval_set=[(X_valid, y_valid)], 
            callbacks=[lgb.early_stopping(50)]
        )
        
        preds = model.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, preds)
        auc_scores.append(auc)
    
    mean_auc = np.mean(auc_scores)
    print(f"[Optuna] Trial {trial.number}: AUC={mean_auc:.4f}")
    return mean_auc


def find_best_lgbm_params(X, y, class_weights):
    """ Optuna를 실행하여 LightGBM 최적 하이퍼파라미터 탐색 """
    print("\nOptuna LightGBM 하이퍼파라미터 탐색 시작...")
    study = optuna.create_study(direction="maximize")
    
    study.optimize(lambda trial: objective_lgbm(trial, X, y, class_weights, n_splits=10), n_trials=50)
    
    print("\nBest trial:", study.best_trial.value)
    print("Best params:", study.best_trial.params)
    return study.best_params


def train_and_save_lgbm_models(X, y, best_params, n_splits=10):
    """ 10-Fold 교차 검증을 실행하고 각 Fold별 LightGBM 모델을 저장 """
    print("\n10-Fold LightGBM 교차 검증 시작...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED) 
    validation_scores = []
    
    for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X), start=1):
        print(f"\nFold {fold_idx}/{n_splits} 학습 중...")
        start_time = time.time()
        
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        # 클래스 가중치 설정
        cw = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: w for i, w in enumerate(cw)}
        
        model = lgb.LGBMClassifier(**best_params, class_weight=class_weight_dict, random_state=SEED) 
        model.fit(
            X_train, 
            y_train, 
            eval_set=[(X_valid, y_valid)], 
            callbacks=[lgb.early_stopping(50)]
        )
        
        preds = model.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, preds)
        validation_scores.append({"fold": fold_idx, "auc": auc})
        
        print(f"Fold {fold_idx} 완료! AUC: {auc:.4f} (소요 시간: {time.time() - start_time:.2f}초)")
        model_path = os.path.join(MODEL_DIR, f"lightgbm_fold_{fold_idx}.txt")
        model.booster_.save_model(model_path)
        print(f"모델 저장 완료: {model_path}")
    
    df_scores = pd.DataFrame(validation_scores)
    df_scores.to_csv(os.path.join(MODEL_DIR, "validation_scores.csv"), index=False)
    print("\n전체 10-Fold Validation AUC 평균: {:.4f} ± {:.4f}".format(
        np.mean(df_scores["auc"]), np.std(df_scores["auc"])
    ))


def main():
    X, y, class_weights = load_data()
    best_lgbm_params = find_best_lgbm_params(X, y, class_weights)
    train_and_save_lgbm_models(X, y, best_lgbm_params)


if __name__ == "__main__":
    main()

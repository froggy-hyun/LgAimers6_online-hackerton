import numpy as np
import pandas as pd
import os
import lightgbm as lgb

# 모델 저장 폴더
MODEL_DIR = "./weight/lightgbm"
N_FOLDS = 10  # 저장된 Fold 개수

def load_test_data():
    """ 테스트 데이터 로드 """
    df_test = pd.read_csv("../data/processed_test_ver2.csv")
    test_ids = df_test["ID"].values  # ID 컬럼 보관
    X_test = df_test.drop(columns=["ID"])  # 모델 입력 데이터
    return X_test, test_ids

def predict():
    """ 10개의 Fold 모델을 활용한 앙상블 예측 """
    X_test, test_ids = load_test_data()
    fold_preds = []

    for fold_idx in range(1, N_FOLDS + 1):
        model_path = os.path.join(MODEL_DIR, f"lightgbm_fold_{fold_idx}.txt")
        print(f"Loading model: {model_path}")

        model = lgb.Booster(model_file=model_path)  # LightGBM 모델 로드

        preds = model.predict(X_test)  # LGBM은 `predict_proba` 없이 `predict` 사용
        fold_preds.append(preds)

    final_preds = np.mean(fold_preds, axis=0)  # 10개 모델 예측 평균(앙상블)

    submission = pd.DataFrame({"ID": test_ids, "Prediction": final_preds})
    submission.to_csv("submission_lightgbm.csv", index=False)
    print("Submission file saved as submission_lightgbm.csv")

if __name__ == "__main__":
    predict()

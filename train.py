import argparse
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_data(csv_path: str):
    df = pd.read_csv(csv_path)

    sort_cols = [
            "game_date",
            "game_id",
            "inning",
            "home_or_away",
            "block_no",
            "option_idx",
            "pitch_num",
            ]
    existing_sort_cols = [c for c in sort_cols if c in df.columns]
    df = df.sort_values(existing_sort_cols).reset_index(drop=True)
    return df

def make_next_pitch_dataset(df: pd.DataFrame):
    df = df.copy()
    df["pitcher_id"] = df["pitcher_id"].astype(str)
    df["game_id"] = df["game_id"].astype(str)
    df["batter_id"] = df["batter_id"].astype(str)
    #pitch_type을 한 칸 위로 이동시켜 next_pitch_type 생성
    df["next_pitch_type"] = (df.groupby(["game_id", "pitcher_id"])["pitch_type"].shift(-1))
    #Define how many pitches to see
    group_cols = ["game_id", "pitcher_id"]
    history_n = 3

    for i in range(1, history_n + 1):
        df[f"prev{i}_pitch_type"] = df.groupby(group_cols)["pitch_type"].shift(i)
        df[f"prev{i}_pitch_result"] = df.groupby(group_cols)["pitch_result"].shift(i)
        df[f"prev{i}_speed_kmh"] = df.groupby(group_cols)["speed_kmh"].shift(i)
        df[f"prev{i}_balls_after"] = df.groupby(group_cols)["balls_after"].shift(i)
        df[f"prev{i}_strikes_after"] = df.groupby(group_cols)["strikes_after"].shift(i)

    #count features
    df["count_state"] = (
        df["balls_after"].fillna(-1).astype(int).astype(str)
        + "-"
        + df["strikes_after"].fillna(-1).astype(int).astype(str)
    )

    df["is_first_pitch"] = (df["pitch_num"] == 1).astype(int)
    df["is_two_strike"] = (df["strikes_after"] >= 2).astype(int)
    df["is_three_ball"] = (df["balls_after"] >= 3).astype(int)
    pa_group_cols = ["game_id", "pitcher_id", "block_no"]

    for i in range(1, history_n + 1):
        df[f"pa_prev{i}_pitch_type"] = df.groupby(pa_group_cols)["pitch_type"].shift(i)
        df[f"pa_prev{i}_pitch_result"] = df.groupby(pa_group_cols)["pitch_result"].shift(i)
    
    # history가 없는 첫 투구들은 NONE으로 표시
    for i in range(1, history_n + 1):
        df[f"prev{i}_pitch_type"] = df[f"prev{i}_pitch_type"].fillna("NONE")
        df[f"prev{i}_pitch_result"] = df[f"prev{i}_pitch_result"].fillna("NONE")
    
    #다음 투구,현재 구종이 없는 row 제거
    df = df.dropna(subset=["next_pitch_type"])
    df = df.dropna(subset=["pitch_type"])
    return df

def train_test_split_by_time(df: pd.DataFrame, test_ratio: float = 0.2):
    df = df.sort_values([
        c for c in [
            "game_date",
            "game_id",
            "inning",
            "home_or_away",
            "block_no",
            "option_idx",
            "pitch_num",
            ]
        if c in df.columns ]).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    return train_df, test_df

def build_logistic():
    categorical_features = [
            "pitcher_id",
            "batter_id",
            "pitch_type",
            "base1",
            "base2",
            "base3",
            "home_or_away",
            "pitch_result",
            "count_state",
            "prev1_pitch_type",
            "prev2_pitch_type",
            "prev3_pitch_type",
            "prev1_pitch_result",
            "prev2_pitch_result",
            "prev3_pitch_result",
            ]
    """
            "pa_prev1_pitch_type",
            "pa_prev2_pitch_type",
            "pa_prev3_pitch_type",
            "pa_prev1_pitch_result",
            "pa_prev2_pitch_result",
            "pa_prev3_pitch_result",
    """
    numeric_features = [
            "inning",
            "pitch_num",
            "speed_kmh",
            "balls_after",
            "strikes_after",
            "outs",
            "home_score",
            "away_score",
            "is_first_pitch",
            "is_two_strike",
            "is_three_ball",
            "prev1_speed_kmh",
            "prev2_speed_kmh",
            "prev3_speed_kmh",
            "prev1_balls_after",
            "prev2_balls_after",
            "prev3_balls_after",
            "prev1_strikes_after",
            "prev2_strikes_after",
            "prev3_strikes_after",
            ]

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features),
        ],
        remainder="drop",
    )
    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
        C=1.0
    )
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return clf, categorical_features + numeric_features


def top_k_accuracy(model, X, y, k=2):
    proba = model.predict_proba(X)
    classes = model.classes_

    top_k_idx = np.argsort(proba, axis=1)[:, -k:]
    top_k_preds = classes[top_k_idx]

    correct = [
        true in preds
        for true, preds in zip(y, top_k_preds)
    ]

    return np.mean(correct)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="pitch csv file path")
    args = parser.parse_args()

    df = load_data(args.csv)
    df = make_next_pitch_dataset(df)

    if len(df) < 30:
        print("데이터가 너무 적습니다. 더 많은 경기 데이터가 필요합니다.")
        print("usable rows:", len(df))
        return
    
    print("전체 학습 가능 row:", len(df))
    print("\nnext_pitch_type 분포:")
    print(df["next_pitch_type"].value_counts())

    train_df, test_df = train_test_split_by_time(df, test_ratio=0.2)

    clf, feature_cols = build_logistic()
    
    X_train = train_df[feature_cols]
    y_train = train_df["next_pitch_type"]

    X_test = test_df[feature_cols]
    y_test = test_df["next_pitch_type"]
    
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    #debug
    """
    print(X_train[[
        "pitch_type",
        "prev1_pitch_type",
        "prev2_pitch_type",
        "prev3_pitch_type",
    ]].head(20))
    """
    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")
    weighted_f1 = f1_score(y_test, pred, average="weighted")

    print("\n===== Evaluation =====")
    print("Accuracy:", acc)
    print("Macro F1:", macro_f1)
    print("Weighted F1:", weighted_f1)

    print("\n===== Classification Report =====")
    print(classification_report(y_test, pred, zero_division=0))

    print("\n===== Confusion Matrix =====")
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, pred, labels=labels)

    cm_df = pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
    print(cm_df)

    top2_acc = top_k_accuracy(clf, X_test, y_test, k=2)
    print("===== Top-2 Accuracy =====")
    print(top2_acc)
    # 예측 결과 일부 확인
    result = test_df.copy()
    result["pred_next_pitch_type"] = pred

    show_cols = [
        "game_date",
        "game_id",
        "inning",
        "batter_name",
        "pitch_type",
        "pitch_result",
        "speed_kmh",
        "balls_after",
        "strikes_after",
        "next_pitch_type",
        "pred_next_pitch_type",
    ]

    show_cols = [c for c in show_cols if c in result.columns]
    """
    print("\n===== Prediction Samples =====")
    print(result[show_cols].head(30).to_string(index=False))
    """

if __name__ == "__main__":
    main()

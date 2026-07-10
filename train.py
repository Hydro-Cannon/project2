import argparse
import pandas as pd

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
    df["next_pitch_type"] = (df.groupby(["game_id", "pitcher_id"])["pitch_type"].shift(-1))
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
            ]
    numeric_features = [
            "inning",
            "pitch_num",
            "speed_kmh",
            "balls_after",
            "strikes_after",
            "outs",
            "home_score",
            "away_score",
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
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return clf, categorical_features + numeric_features


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

    print("\n===== Prediction Samples =====")
    print(result[show_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()

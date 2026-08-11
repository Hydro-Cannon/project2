import argparse
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, log_loss
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
    pa_group_cols = ["game_id", "pitcher_id", "block_no"]
    #pitch_type을 한 칸 위로 이동시켜 next_pitch_type 생성
    df["next_pitch_type"] = (df.groupby(pa_group_cols)["pitch_type"].shift(-1))
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
        df[f"prev{i}_pitch_type"] = (
            df[f"prev{i}_pitch_type"].fillna("NONE")
        )
        df[f"prev{i}_pitch_result"] = (
            df[f"prev{i}_pitch_result"].fillna("NONE")
        )

        df[f"pa_prev{i}_pitch_type"] = (
            df[f"pa_prev{i}_pitch_type"].fillna("NONE")
        )
        df[f"pa_prev{i}_pitch_result"] = (
            df[f"pa_prev{i}_pitch_result"].fillna("NONE")
        )
    # sequence of 2 pitches
    df["prev2_pitch_sequence"] = (
            df["prev2_pitch_type"].astype(str)
            + "->"
            + df["prev1_pitch_type"].astype(str)
            )
    # Count interaction
    df["pitch_count"] = (
            df["pitch_type"].astype(str)
            + "_"
            + df["count_state"].astype(str)
            )
    # previous type + constant count
    df["prev_pitch_count"] = (
            df["prev1_pitch_type"].astype(str)
            + "_"
            + df["count_state"].astype(str)
            )
    """
    for i in range(1, history_n + 1):
        df[f"prev{i}_pitch_type"] = df[f"prev{i}_pitch_type"].fillna("NONE")
        df[f"prev{i}_pitch_result"] = df[f"prev{i}_pitch_result"].fillna("NONE")
    """
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

def train_test_split_by_game(df: pd.DataFrame, test_ratio: float = 0.2):
    df = df.copy()
    df["game_id"] = df["game_id"].astype(str)

    games = (
            df[["game_date", "game_id"]].drop_duplicates().sort_values(["game_date", "game_id"]).reset_index(drop=True)
            )
    n_games = len(games)
    split_idx = int(n_games * (1 - test_ratio))
    split_idx = max(1, min(split_idx, n_games - 1))

    train_games = set(games.iloc[:split_idx]["game_id"])
    test_games = set(games.iloc[split_idx:]["game_id"])

    train_df = df[df["game_id"].isin(train_games)].copy()
    test_df = df[df["game_id"].isin(test_games)].copy()

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

    train_df = (
            train_df.sort_values(existing_sort_cols).reset_index(drop=True)
            )
    test_df = (
            test_df.sort_values(existing_sort_cols).reset_index(drop=True)
            )
    #debug
    print("\n===== Game-based Train/Test Split =====")
    print("Total games :", n_games)
    print("Train games :", len(train_games))
    print("Test games  :", len(test_games))
    print("Train rows  :", len(train_df))
    print("Test rows   :", len(test_df))

    print("\nTrain period:")
    print(
        train_df["game_date"].min(),
        "~",
        train_df["game_date"].max()
    )

    print("Test period:")
    print(
        test_df["game_date"].min(),
        "~",
        test_df["game_date"].max()
    )

    #overlap check
    overlap = train_games & test_games
    assert len(overlap) == 0, f"Train/Test game overlap: {overlap}"

    return train_df, test_df

def train_val_test_split_by_game(
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        ):
    df = df.copy()
    df["game_id"] = df["game_id"].astype(str)

    # 경기 목록을 시간순으로 정렬
    games = (
        df[["game_date", "game_id"]]
        .drop_duplicates()
        .sort_values(["game_date", "game_id"])
        .reset_index(drop=True)
    )

    n_games = len(games)

    train_end = int(n_games * train_ratio)
    val_end = int(n_games * (train_ratio + val_ratio))

    train_games = set(games.iloc[:train_end]["game_id"])
    val_games = set(games.iloc[train_end:val_end]["game_id"])
    test_games = set(games.iloc[val_end:]["game_id"])

    train_df = df[df["game_id"].isin(train_games)].copy()
    val_df = df[df["game_id"].isin(val_games)].copy()
    test_df = df[df["game_id"].isin(test_games)].copy()

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

    train_df = train_df.sort_values(existing_sort_cols).reset_index(drop=True)
    val_df = val_df.sort_values(existing_sort_cols).reset_index(drop=True)
    test_df = test_df.sort_values(existing_sort_cols).reset_index(drop=True)

    # overlap 검사
    assert len(train_games & val_games) == 0
    assert len(train_games & test_games) == 0
    assert len(val_games & test_games) == 0

    print("\n===== Game-based Train / Validation / Test Split =====")
    print("Total games :", n_games)
    print("Train games :", len(train_games))
    print("Val games   :", len(val_games))
    print("Test games  :", len(test_games))

    print("\nTrain rows :", len(train_df))
    print("Val rows   :", len(val_df))
    print("Test rows  :", len(test_df))

    print("\nTrain period:")
    print(train_df["game_date"].min(), "~", train_df["game_date"].max())

    print("Validation period:")
    print(val_df["game_date"].min(), "~", val_df["game_date"].max())

    print("Test period:")
    print(test_df["game_date"].min(), "~", test_df["game_date"].max())

    return train_df, val_df, test_df

def build_logistic(C=1.0, class_weight = None):
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
            "pa_prev1_pitch_type",
            "pa_prev2_pitch_type",
            "pa_prev3_pitch_type",
            "pa_prev1_pitch_result",
            "pa_prev2_pitch_result",
            "pa_prev3_pitch_result",
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
    """
    weights_to_try = [
        None,
        {
            "직구": 1.0,
            "체인지업": 1.0,
            "슬라이더": 1.2,
            "커브": 1.0,
        },

        {
            "직구": 1.0,
            "체인지업": 1.1,
            "슬라이더": 1.3,
            "커브": 1.0,
        },

        {
            "직구": 1.0,
            "체인지업": 1.1,
            "슬라이더": 1.4,
            "커브": 1.1,
        },
    ]
    """
    model = LogisticRegression(
        max_iter=3000,
        class_weight=class_weight,
        solver="lbfgs",
        C=C
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

    #train_df, test_df = train_test_split_by_time(df, test_ratio=0.2)

    train_df, val_df, test_df = train_val_test_split_by_game(df, train_ratio=0.7, val_ratio=0.15,)
    """
    clf, feature_cols = build_logistic()
    
    X_train = train_df[feature_cols]
    y_train = train_df["next_pitch_type"]
    
    X_val = val_df[feature_cols]
    y_val = val_df["next_pitch_type"]
    
    X_test = test_df[feature_cols]
    y_test = test_df["next_pitch_type"]
    """
    C_values = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    
    best_C = None
    best_val_acc = -1

    print("\n===== C Tuning =====")

    for C in C_values:

        clf, feature_cols = build_logistic(
            C=C,
            class_weight=None
        )

        X_train = train_df[feature_cols]
        y_train = train_df["next_pitch_type"]

        X_val = val_df[feature_cols]
        y_val = val_df["next_pitch_type"]

        clf.fit(X_train, y_train)

        train_pred = clf.predict(X_train)
        val_pred = clf.predict(X_val)

        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)

        val_macro_f1 = f1_score(
            y_val,
            val_pred,
            average="macro"
            )

        val_proba = clf.predict_proba(X_val)

        val_loss = log_loss(
            y_val,
            val_proba,
            labels=clf.classes_
        )

        print(
            f"C={C:<5} "
            f"Train Acc={train_acc:.4f} "
            f"Val Acc={val_acc:.4f} "
            f"Val F1={val_macro_f1:.4f} "
            f"Val Loss={val_loss:.4f}"
        )   

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_C = C


    print("\nBest C:", best_C)
    print("Best Validation Accuracy:", best_val_acc)
    """
    clf.fit(X_train, y_train)
    
    train_pred = clf.predict(X_train)
    pred = clf.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, pred)
    train_error = 1 - train_acc
    test_error = 1- test_acc
    print("\n===== Train / Test Error =====")
    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Train Error    : {train_error:.4f}")
    print(f"Test Accuracy  : {test_acc:.4f}")
    print(f"Test Error     : {test_error:.4f}")
    print(f"Error Gap      : {test_error - train_error:.4f}")
    train_proba = clf.predict_proba(X_train)
    test_proba = clf.predict_proba(X_test)

    train_loss = log_loss(y_train, train_proba, labels=clf.classes_)
    test_loss = log_loss(y_test, test_proba, labels=clf.classes_)

    print("\n===== Log Loss =====")
    print(f"Train Loss : {train_loss:.4f}")
    print(f"Test Loss  : {test_loss:.4f}")
    """
    #debug
    """
    print(X_train[[
        "pitch_type",
        "prev1_pitch_type",
        "prev2_pitch_type",
        "prev3_pitch_type",
    ]].head(20))
    """
    train_df = pd.concat([train_df, val_df], ignore_index=True)
    X_train = train_df[feature_cols]
    y_train = train_df["next_pitch_type"]
    X_test = test_df[feature_cols]
    y_test = test_df["next_pitch_type"]
    clf, feature_cols = build_logistic(C=best_C, class_weight=None)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")
    weighted_f1 = f1_score(y_test, pred, average="weighted")

    print("\n===== Final Evaluation =====")
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

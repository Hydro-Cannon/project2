import argparse
from itertools import product

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm.auto import tqdm


SORT_COLS = [
    "game_date",
    "game_id",
    "inning",
    "home_or_away",
    "block_no",
    "option_idx",
    "pitch_num",
]


def sort_pitch_rows(df: pd.DataFrame) -> pd.DataFrame:
    existing_cols = [col for col in SORT_COLS if col in df.columns]
    return df.sort_values(existing_cols).reset_index(drop=True)


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return sort_pitch_rows(df)


def make_next_pitch_dataset(
    df: pd.DataFrame,
    history_n: int = 2,
) -> pd.DataFrame:
    """현재 투구까지의 정보로 같은 타석의 다음 구종을 예측하는 데이터셋을 만든다."""
    if history_n < 1:
        raise ValueError("history_n은 1 이상이어야 합니다.")

    df = sort_pitch_rows(df.copy())

    for col in ["pitcher_id", "game_id", "batter_id"]:
        df[col] = df[col].astype(str)

    pa_group_cols = ["game_id", "pitcher_id", "block_no"]
    game_group_cols = ["game_id", "pitcher_id"]

    # 이 값들은 마지막 투구를 제거하기 전에 계산해야 한다.
    df["game_pitch_number"] = df.groupby(game_group_cols).cumcount() + 1

    # 이 경기에서 같은 타자를 몇 번째 타석에서 상대하는지 계산한다.
    pa_order = (
        df[["game_id", "pitcher_id", "batter_id", "block_no"]]
        .drop_duplicates()
        .sort_values(["game_id", "pitcher_id", "block_no"])
        .reset_index(drop=True)
    )
    pa_order["times_faced_batter"] = (
        pa_order.groupby(["game_id", "pitcher_id", "batter_id"]).cumcount()
        + 1
    )
    df = df.merge(
        pa_order,
        on=["game_id", "pitcher_id", "batter_id", "block_no"],
        how="left",
        validate="many_to_one",
    )
    df = sort_pitch_rows(df)

    # 같은 타석 안에서만 다음 투구를 정답으로 만든다.
    df["next_pitch_type"] = df.groupby(pa_group_cols)["pitch_type"].shift(-1)

    # 경기 전체 흐름의 최근 투구 기록
    for i in range(1, history_n + 1):
        df[f"prev{i}_pitch_type"] = df.groupby(game_group_cols)[
            "pitch_type"
        ].shift(i)
        df[f"prev{i}_pitch_result"] = df.groupby(game_group_cols)[
            "pitch_result"
        ].shift(i)
        df[f"prev{i}_speed_kmh"] = df.groupby(game_group_cols)[
            "speed_kmh"
        ].shift(i)
        df[f"prev{i}_balls_after"] = df.groupby(game_group_cols)[
            "balls_after"
        ].shift(i)
        df[f"prev{i}_strikes_after"] = df.groupby(game_group_cols)[
            "strikes_after"
        ].shift(i)

    # 같은 타석 안에서의 최근 투구 기록
    for i in range(1, history_n + 1):
        df[f"pa_prev{i}_pitch_type"] = df.groupby(pa_group_cols)[
            "pitch_type"
        ].shift(i)
        df[f"pa_prev{i}_pitch_result"] = df.groupby(pa_group_cols)[
            "pitch_result"
        ].shift(i)

    df["count_state"] = (
        df["balls_after"].fillna(-1).astype(int).astype(str)
        + "-"
        + df["strikes_after"].fillna(-1).astype(int).astype(str)
    )

    df["is_first_pitch"] = (df["pitch_num"] == 1).astype(int)
    df["is_two_strike"] = (df["strikes_after"] >= 2).astype(int)
    df["is_three_ball"] = (df["balls_after"] >= 3).astype(int)

    # 빈 베이스는 NaN이 아니라 0으로 저장되므로 notna()를 사용하면 안 된다.
    for col in ["base1", "base2", "base3"]:
        if col not in df.columns:
            df[col] = 0

    df["runners_on_base"] = (
        df["base1"].fillna(0).ne(0).astype(int)
        + df["base2"].fillna(0).ne(0).astype(int)
        + df["base3"].fillna(0).ne(0).astype(int)
    )
    df["is_risp"] = (
        df["base2"].fillna(0).ne(0)
        | df["base3"].fillna(0).ne(0)
    ).astype(int)

    # home_or_away=0이면 홈 투수가 초 공격을 막고, 1이면 원정 투수가
    # 말 공격을 막는다. 어느 경기장이든 양수는 투수 팀의 리드를 뜻한다.
    pitcher_team_sign = df["home_or_away"].map({0: 1, 1: -1})
    df["pitcher_score_diff"] = (
        df["home_score"] - df["away_score"]
    ) * pitcher_team_sign

    for i in range(1, history_n + 1):
        for suffix in ["pitch_type", "pitch_result"]:
            df[f"prev{i}_{suffix}"] = df[f"prev{i}_{suffix}"].fillna("NONE")
            df[f"pa_prev{i}_{suffix}"] = df[f"pa_prev{i}_{suffix}"].fillna(
                "NONE"
            )

    # 로지스틱 회귀가 중요한 비선형 조합을 직접 학습할 수 있게 만든다.
    prev2_type = (
        df["prev2_pitch_type"]
        if history_n >= 2
        else pd.Series("NONE", index=df.index)
    )
    df["prev2_pitch_sequence"] = (
        prev2_type.astype(str) + "->" + df["prev1_pitch_type"].astype(str)
    )
    df["pitch_count"] = (
        df["pitch_type"].astype(str) + "_" + df["count_state"].astype(str)
    )
    df["prev_pitch_count"] = (
        df["prev1_pitch_type"].astype(str)
        + "_"
        + df["count_state"].astype(str)
    )

    df["speed_diff_prev1"] = df["speed_kmh"] - df["prev1_speed_kmh"]

    tracking_cols = [
        "pts_crossPlateX",
        "pts_crossPlateY",
        "pts_topSz",
        "pts_bottomSz",
        "pts_vx0",
        "pts_vy0",
        "pts_vz0",
        "pts_ax",
        "pts_ay",
        "pts_az",
    ]
    for col in tracking_cols:
        if col not in df.columns:
            df[col] = np.nan

    if "pts_stance" not in df.columns:
        df["pts_stance"] = "UNKNOWN"
    else:
        df["pts_stance"] = df["pts_stance"].fillna("UNKNOWN")

    df = df.dropna(subset=["next_pitch_type", "pitch_type"])
    return df.reset_index(drop=True)


def get_ordered_games(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[["game_date", "game_id"]]
        .drop_duplicates()
        .sort_values(["game_date", "game_id"])
        .reset_index(drop=True)
    )


def split_dev_test_by_game(
    df: pd.DataFrame,
    test_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """마지막 경기들을 최종 테스트로 완전히 분리한다."""
    games = get_ordered_games(df)
    n_games = len(games)

    if n_games < 3:
        raise ValueError("경기 단위 분할을 하려면 최소 3경기가 필요합니다.")

    n_test = max(1, int(round(n_games * test_ratio)))
    n_test = min(n_test, n_games - 2)

    dev_games = set(games.iloc[:-n_test]["game_id"].astype(str))
    test_games = set(games.iloc[-n_test:]["game_id"].astype(str))

    dev_df = sort_pitch_rows(df[df["game_id"].isin(dev_games)].copy())
    test_df = sort_pitch_rows(df[df["game_id"].isin(test_games)].copy())

    assert not (dev_games & test_games)

    print("\n===== Development / Final Test Split =====")
    print("Total games:", n_games)
    print("Development games:", len(dev_games))
    print("Final test games:", len(test_games))
    print("Development rows:", len(dev_df))
    print("Final test rows:", len(test_df))
    print(
        "Development period:",
        dev_df["game_date"].min(),
        "~",
        dev_df["game_date"].max(),
    )
    print(
        "Final test period:",
        test_df["game_date"].min(),
        "~",
        test_df["game_date"].max(),
    )

    return dev_df, test_df


def make_walk_forward_folds(
    dev_df: pd.DataFrame,
    n_splits: int = 4,
    val_games_per_fold: int = 6,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """과거 경기로 학습하고 바로 다음 경기들로 검증하는 expanding-window fold."""
    games = get_ordered_games(dev_df)
    n_games = len(games)

    max_splits = (n_games - 2) // val_games_per_fold
    n_splits = min(n_splits, max_splits)

    if n_splits < 2:
        raise ValueError(
            "walk-forward 검증을 위한 경기가 부족합니다. "
            "n_splits 또는 val_games_per_fold를 줄이세요."
        )

    first_val_start = n_games - n_splits * val_games_per_fold
    folds = []

    print("\n===== Walk-forward Folds =====")
    for fold_idx in range(n_splits):
        val_start = first_val_start + fold_idx * val_games_per_fold
        val_end = val_start + val_games_per_fold

        train_game_ids = set(games.iloc[:val_start]["game_id"].astype(str))
        val_game_ids = set(
            games.iloc[val_start:val_end]["game_id"].astype(str)
        )

        train_df = sort_pitch_rows(
            dev_df[dev_df["game_id"].isin(train_game_ids)].copy()
        )
        val_df = sort_pitch_rows(
            dev_df[dev_df["game_id"].isin(val_game_ids)].copy()
        )

        assert not (train_game_ids & val_game_ids)
        folds.append((train_df, val_df))

        print(
            f"Fold {fold_idx + 1}: "
            f"train={len(train_game_ids)} games/{len(train_df)} rows, "
            f"val={len(val_game_ids)} games/{len(val_df)} rows, "
            f"val period={val_df['game_date'].min()}~{val_df['game_date'].max()}"
        )

    return folds


def get_feature_columns(
    history_n: int,
    use_batter_id: bool = False,
    use_tracking: bool = False,
) -> tuple[list[str], list[str]]:
    categorical_features = [
        "pitch_type",
        "home_or_away",
        "pitch_result",
        "count_state",
        "pts_stance",
        "pitch_count",
        "prev_pitch_count",
        "prev2_pitch_sequence",
    ]

    if use_batter_id:
        categorical_features.insert(0, "batter_id")

    for i in range(1, history_n + 1):
        categorical_features.extend(
            [
                f"prev{i}_pitch_type",
                f"prev{i}_pitch_result",
                f"pa_prev{i}_pitch_type",
                f"pa_prev{i}_pitch_result",
            ]
        )

    numeric_features = [
        "inning",
        "pitch_num",
        "game_pitch_number",
        "speed_kmh",
        "balls_after",
        "strikes_after",
        "outs",
        "home_score",
        "away_score",
        "pitcher_score_diff",
        "runners_on_base",
        "is_risp",
        "times_faced_batter",
        "is_first_pitch",
        "is_two_strike",
        "is_three_ball",
        "speed_diff_prev1",
    ]

    for i in range(1, history_n + 1):
        numeric_features.extend(
            [
                f"prev{i}_speed_kmh",
                f"prev{i}_balls_after",
                f"prev{i}_strikes_after",
            ]
        )

    if use_tracking:
        numeric_features.extend(
            [
                "pts_crossPlateX",
                "pts_topSz",
                "pts_vy0",
                "pts_ay",
            ]
        )

    return categorical_features, numeric_features


def build_logistic(
    categorical_features: list[str],
    numeric_features: list[str],
    C: float = 1.0,
    class_weight: dict | None = None,
) -> Pipeline:
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
        class_weight=class_weight,
        solver="lbfgs",
        C=C,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def make_soft_class_weight(
    y: pd.Series,
    alpha: float,
) -> dict | None:
    """alpha=0은 가중치 없음, alpha=1은 sklearn의 balanced와 같다."""
    if alpha <= 0:
        return None

    counts = y.value_counts()
    n_samples = len(y)
    n_classes = len(counts)

    return {
        label: (n_samples / (n_classes * count)) ** alpha
        for label, count in counts.items()
    }


def make_recency_weight(
    game_dates: pd.Series,
    half_life_days: int | None,
) -> np.ndarray | None:
    if half_life_days is None or half_life_days <= 0:
        return None

    dates = pd.to_datetime(
        game_dates.astype(str),
        format="%Y%m%d",
        errors="raise",
    )
    age_days = (dates.max() - dates).dt.days.to_numpy()
    weights = np.power(0.5, age_days / half_life_days)

    # 평균을 1로 맞춰 정규화 강도 C의 의미가 크게 변하지 않게 한다.
    return weights / weights.mean()


def calculate_metrics(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float]:
    pred = model.predict(X)
    proba = model.predict_proba(X)

    accuracy = accuracy_score(y, pred)
    macro_f1 = f1_score(y, pred, average="macro", zero_division=0)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": f1_score(
            y,
            pred,
            average="weighted",
            zero_division=0,
        ),
        "log_loss": log_loss(y, proba, labels=model.classes_),
        "combined": 0.5 * accuracy + 0.5 * macro_f1,
    }


def tune_hyperparameters(
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    categorical_features: list[str],
    numeric_features: list[str],
    selection_metric: str = "combined",
) -> tuple[dict, pd.DataFrame]:
    feature_cols = categorical_features + numeric_features

    C_values = [0.001, 0.003, 0.01, 0.03, 0.1]
    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    half_life_values = [None, 180, 365]

    results = []
    configs = list(product(C_values, alpha_values, half_life_values))

    print("\n===== Walk-forward Hyperparameter Search =====")
    print("Configurations:", len(configs))
    print("Models to fit:", len(configs) * len(folds))
    print("Selection metric:", selection_metric)

    for C, alpha, half_life in tqdm(
        configs,
        desc="Hyperparameter search",
        unit="config",
    ):
        fold_metrics = []

        for train_df, val_df in folds:
            class_weight = make_soft_class_weight(
                train_df["next_pitch_type"],
                alpha,
            )
            sample_weight = make_recency_weight(
                train_df["game_date"],
                half_life,
            )

            model = build_logistic(
                categorical_features,
                numeric_features,
                C=C,
                class_weight=class_weight,
            )

            fit_params = {}
            if sample_weight is not None:
                fit_params["model__sample_weight"] = sample_weight

            model.fit(
                train_df[feature_cols],
                train_df["next_pitch_type"],
                **fit_params,
            )
            fold_metrics.append(
                calculate_metrics(
                    model,
                    val_df[feature_cols],
                    val_df["next_pitch_type"],
                )
            )

        row = {
            "C": C,
            "alpha": alpha,
            "half_life_days": 0 if half_life is None else half_life,
        }
        for metric in [
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "log_loss",
            "combined",
        ]:
            values = [m[metric] for m in fold_metrics]
            row[f"mean_{metric}"] = float(np.mean(values))
            row[f"std_{metric}"] = float(np.std(values))

        results.append(row)

    result_df = pd.DataFrame(results)
    score_col = f"mean_{selection_metric}"
    ascending = selection_metric == "log_loss"
    result_df = result_df.sort_values(
        [score_col, "mean_accuracy", "mean_macro_f1"],
        ascending=[ascending, False, False],
    ).reset_index(drop=True)

    best = result_df.iloc[0].to_dict()

    print("\n===== Top 10 Configurations =====")
    print(
        result_df[
            [
                "C",
                "alpha",
                "half_life_days",
                "mean_accuracy",
                "mean_macro_f1",
                "mean_log_loss",
                "mean_combined",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    return best, result_df


def top_k_accuracy(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 2,
) -> float:
    proba = model.predict_proba(X)
    classes = model.classes_
    top_k_idx = np.argsort(proba, axis=1)[:, -k:]
    top_k_preds = classes[top_k_idx]
    return float(
        np.mean([true in preds for true, preds in zip(y, top_k_preds)])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="pitch CSV file path")
    parser.add_argument(
        "--history-n",
        type=int,
        default=2,
        help="사용할 최근 투구 수 (기본값: 2)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="마지막 테스트 경기 비율 (기본값: 0.15)",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=4,
        help="walk-forward fold 수 (기본값: 4)",
    )
    parser.add_argument(
        "--val-games",
        type=int,
        default=6,
        help="fold마다 사용할 검증 경기 수 (기본값: 6)",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["combined", "accuracy", "macro_f1", "log_loss"],
        default="combined",
        help="모델 선택 지표 (기본값: combined)",
    )
    parser.add_argument(
        "--use-batter-id",
        action="store_true",
        help="고차원 batter_id 특징을 사용",
    )
    parser.add_argument(
        "--use-tracking",
        action="store_true",
        help="일부 PTS 트래킹 특징을 사용",
    )
    args = parser.parse_args()

    df = load_data(args.csv)
    df = make_next_pitch_dataset(df, history_n=args.history_n)

    if len(df) < 30:
        print("데이터가 너무 적습니다. 더 많은 경기 데이터가 필요합니다.")
        print("Usable rows:", len(df))
        return

    print("전체 학습 가능 row:", len(df))
    print("\nnext_pitch_type 분포:")
    print(df["next_pitch_type"].value_counts())
    print("\ntimes_faced_batter 분포:")
    print(df["times_faced_batter"].value_counts().sort_index())
    print("\nrunners_on_base 분포:")
    print(df["runners_on_base"].value_counts().sort_index())

    dev_df, test_df = split_dev_test_by_game(
        df,
        test_ratio=args.test_ratio,
    )
    folds = make_walk_forward_folds(
        dev_df,
        n_splits=args.cv_splits,
        val_games_per_fold=args.val_games,
    )

    categorical_features, numeric_features = get_feature_columns(
        history_n=args.history_n,
        use_batter_id=args.use_batter_id,
        use_tracking=args.use_tracking,
    )
    feature_cols = categorical_features + numeric_features

    best, _ = tune_hyperparameters(
        folds,
        categorical_features,
        numeric_features,
        selection_metric=args.selection_metric,
    )

    best_C = float(best["C"])
    best_alpha = float(best["alpha"])
    best_half_life = int(best["half_life_days"])
    if best_half_life == 0:
        best_half_life = None

    print("\n===== Selected Configuration =====")
    print("C:", best_C)
    print("Class-weight alpha:", best_alpha)
    print("Recency half-life days:", best_half_life)
    print("History length:", args.history_n)
    print("Use batter_id:", args.use_batter_id)
    print("Use tracking:", args.use_tracking)

    final_class_weight = make_soft_class_weight(
        dev_df["next_pitch_type"],
        best_alpha,
    )
    final_sample_weight = make_recency_weight(
        dev_df["game_date"],
        best_half_life,
    )

    final_model = build_logistic(
        categorical_features,
        numeric_features,
        C=best_C,
        class_weight=final_class_weight,
    )

    fit_params = {}
    if final_sample_weight is not None:
        fit_params["model__sample_weight"] = final_sample_weight

    final_model.fit(
        dev_df[feature_cols],
        dev_df["next_pitch_type"],
        **fit_params,
    )

    X_test = test_df[feature_cols]
    y_test = test_df["next_pitch_type"]
    pred = final_model.predict(X_test)
    metrics = calculate_metrics(final_model, X_test, y_test)

    print("\n===== Final Test Evaluation =====")
    print("Accuracy:", metrics["accuracy"])
    print("Macro F1:", metrics["macro_f1"])
    print("Weighted F1:", metrics["weighted_f1"])
    print("Log Loss:", metrics["log_loss"])
    print("Combined Score:", metrics["combined"])
    print("Top-2 Accuracy:", top_k_accuracy(final_model, X_test, y_test, k=2))

    print("\n===== Classification Report =====")
    print(classification_report(y_test, pred, zero_division=0))

    print("\n===== Confusion Matrix =====")
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )
    print(cm_df)


if __name__ == "__main__":
    main()

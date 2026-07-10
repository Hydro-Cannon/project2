# main.py

import os
import pandas as pd

from collector import collect_player_game_ids, load_game_ids, save_game_ids
from parser import collect_pitcher_pitches, save_pitcher_csv


TARGET_PITCHER_ID = "77637"
TARGET_PITCHER_NAME = "양현종"

OUTPUT_CSV = f"kbo_pitcher_{TARGET_PITCHER_NAME}_{TARGET_PITCHER_ID}.csv"


def load_existing_pitch_csv(csv_path: str):
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].astype(str)

    return df


def sort_pitch_df(df: pd.DataFrame):
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

    if existing_sort_cols:
        df = df.sort_values(existing_sort_cols)

    return df.reset_index(drop=True)


def dedupe_pitch_df(df: pd.DataFrame):
    if df.empty:
        return df

    if "pts_pitch_id" in df.columns:
        df = df.drop_duplicates(subset=["game_id", "pts_pitch_id"], keep="last")
    else:
        df = df.drop_duplicates(keep="last")

    return df


def main():
    # 1. 기존 gameId 기록 읽기
    existing_game_ids = load_game_ids(TARGET_PITCHER_ID)
    print("기존 gameId 기록 수:", len(existing_game_ids))

    # 2. collector로 새 gameId 찾기
    collected_game_ids = collect_player_game_ids(TARGET_PITCHER_ID)
    print("collector에서 찾은 gameId 수:", len(collected_game_ids))

    # 3. 기존 기록 + 새 기록 병합 저장
    save_game_ids(TARGET_PITCHER_ID, collected_game_ids)

    all_game_ids = load_game_ids(TARGET_PITCHER_ID)
    print("병합 후 전체 gameId 수:", len(all_game_ids))

    # 4. 기존 CSV 읽기
    existing_df = load_existing_pitch_csv(OUTPUT_CSV)

    if not existing_df.empty and "game_id" in existing_df.columns:
        already_collected_game_ids = set(existing_df["game_id"].astype(str).unique())
    else:
        already_collected_game_ids = set()

    print("이미 CSV에 있는 경기 수:", len(already_collected_game_ids))

    # 5. CSV에 없는 경기만 새로 수집
    new_game_ids = [
        gid for gid in all_game_ids
        if gid not in already_collected_game_ids
    ]

    print("새로 수집할 경기 수:", len(new_game_ids))

    if len(new_game_ids) == 0:
        print("추가로 수집할 경기가 없습니다.")
        print("기존 CSV 유지:", OUTPUT_CSV)
        return

    for gid in new_game_ids:
        print("new game:", gid)

    # 6. 새 경기만 pitch 수집
    new_df = collect_pitcher_pitches(
        game_ids=new_game_ids,
        pitcher_id=TARGET_PITCHER_ID,
        pitcher_name=TARGET_PITCHER_NAME,
        sleep_sec=0.5,
    )

    if new_df.empty:
        print("새로 수집된 투구 데이터가 없습니다.")
        return

    # 7. 기존 CSV + 새 데이터 합치기
    if existing_df.empty:
        result_df = new_df
    else:
        result_df = pd.concat([existing_df, new_df], ignore_index=True)

    # 8. 중복 제거 + 정렬
    result_df = dedupe_pitch_df(result_df)
    result_df = sort_pitch_df(result_df)

    # 9. 같은 CSV에 다시 저장
    save_pitcher_csv(
        df=result_df,
        pitcher_name=TARGET_PITCHER_NAME,
        pitcher_id=TARGET_PITCHER_ID,
        output_path=OUTPUT_CSV,
    )

    print("저장 완료:", OUTPUT_CSV)
    print("전체 투구 수:", len(result_df))
    print("전체 경기 수:", result_df["game_id"].nunique())


if __name__ == "__main__":
    main()

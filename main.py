# main.py

from collector import collect_player_game_ids, save_game_ids
from parser import collect_pitcher_pitches, save_pitcher_csv


TARGET_PITCHER_ID = "77637"
TARGET_PITCHER_NAME = "양현종"


def main():
    # 1. 선수 페이지에서 gameId 수집
    game_ids = collect_player_game_ids(TARGET_PITCHER_ID)
    save_game_ids(TARGET_PITCHER_ID, game_ids)

    print("찾은 경기 수:", len(game_ids))

    # 2. gameId 목록에서 해당 투수 pitch만 수집
    result_df = collect_pitcher_pitches(
        game_ids=game_ids,
        pitcher_id=TARGET_PITCHER_ID,
        sleep_sec=0.5,
    )

    if result_df.empty:
        print("NO PITCHING DATA")
        return

    # 3. CSV 저장
    output_name = save_pitcher_csv(
        df=result_df,
        pitcher_name=TARGET_PITCHER_NAME,
        pitcher_id=TARGET_PITCHER_ID,
    )

    print("Saved:", output_name)
    print("Overall pitches:", len(result_df))

    preview_cols = [
        "game_date",
        "game_id",
        "inning",
        "pitcher_name",
        "batter_name",
        "pitch_num",
        "pitch_result",
        "speed_kmh",
        "pitch_type",
        "balls_after",
        "strikes_after",
        "outs",
    ]

    existing_cols = [c for c in preview_cols if c in result_df.columns]
    print(result_df[existing_cols].head(30))


if __name__ == "__main__":
    main()

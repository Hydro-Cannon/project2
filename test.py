from inning_parser import collect_relay_data_by_innings
from parser import parse_relays_to_dataframe

GAME_ID = "20260701SKHT02026"

relays = collect_relay_data_by_innings(GAME_ID, headless=True)

print("수집 relay 수:", len(relays))

df = parse_relays_to_dataframe(relays)
print("\n이닝별 투구 수:")
print(df.groupby("inning").size().sort_index())
target_df = df[df["pitcher_id"].astype(str) == "77637"].copy()

print("\n양현종 이닝별 투구 수:")
print(target_df.groupby("inning").size().sort_index())

print("\n양현종 2회:")
print(target_df[target_df["inning"] == 2][[
    "inning",
    "pitcher_id",
    "pitcher_name",
    "batter_name",
    "pitch_num",
    "pitch_result",
    "speed_kmh",
    "pitch_type",
    "balls_after",
    "strikes_after",
]].to_string(index=False))

import requests
import pandas as pd

GAME_ID = "20260701SKHT02026"

url = f"https://api-gw.sports.naver.com/schedule/games/{GAME_ID}/relay"

headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}


def get_relay(game_id):
    url = f"https://api-gw.sports.naver.com/schedule/games/{game_id}/relay"
    res = requests.get(url, headers=headers, timeout=10)
    res.raise_for_status()
    data = res.json()
    return data["result"]["textRelayData"]


def build_player_map(relay):
    player_map = {}

    for side in ["home", "away"]:
        for group in ["Lineup", "Entry"]:
            section = relay.get(f"{side}{group}", {})

            for role in ["batter", "pitcher"]:
                for p in section.get(role, []):
                    pcode = p.get("pcode")
                    name = p.get("name")

                    if pcode and name:
                        player_map[str(pcode)] = name

    return player_map


def flatten_dict(d, prefix=""):
    out = {}

    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            out.update(flatten_dict(v, key))
    else:
        out[prefix] = d

    return out


def parse_pitch_rows(relay):
    player_map = build_player_map(relay)
    rows = []

    for block_idx, block in enumerate(relay.get("textRelays", [])):
        inning = block.get("inn")
        home_or_away = block.get("homeOrAway")
        title = block.get("title")
        block_no = block.get("no")

        # ptsOptions를 pitchId 기준 dict로 변환
        pts_map = {}
        for pts in block.get("ptsOptions") or []:
            pitch_id = pts.get("pitchId")
            if pitch_id:
                pts_map[pitch_id] = pts

        for opt_idx, opt in enumerate(block.get("textOptions", [])):
            # pitchNum이 없는 textOption은 "5번타자 카스트로" 같은 설명 줄이므로 제외
            if "pitchNum" not in opt:
                continue

            state = opt.get("currentGameState") or {}

            pitcher_id = str(state.get("pitcher")) if state.get("pitcher") is not None else None
            batter_id = str(state.get("batter")) if state.get("batter") is not None else None

            pts_pitch_id = opt.get("ptsPitchId")
            pts = pts_map.get(pts_pitch_id, {})

            row = {
                "game_id": relay.get("gameId"),
                "block_idx": block_idx,
                "block_no": block_no,
                "option_idx": opt_idx,

                "inning": inning,
                "home_or_away": home_or_away,
                "pa_title": title,

                "pitcher_id": pitcher_id,
                "pitcher_name": player_map.get(pitcher_id),
                "batter_id": batter_id,
                "batter_name": player_map.get(batter_id),

                "pitch_num": opt.get("pitchNum"),
                "pitch_result": opt.get("pitchResult"),
                "pts_pitch_id": pts_pitch_id,

                # 핵심: 구속, 구종
                "speed_kmh": opt.get("speed"),
                "pitch_type": opt.get("stuff"),

                # current count/state
                "balls_after": state.get("ball"),
                "strikes_after": state.get("strike"),
                "outs": state.get("out"),

                "base1": state.get("base1"),
                "base2": state.get("base2"),
                "base3": state.get("base3"),

                "home_score": state.get("homeScore"),
                "away_score": state.get("awayScore"),

                "text": opt.get("text"),
            }

            # ptsOptions의 트래킹 데이터 추가
            for k, v in pts.items():
                row[f"pts_{k}"] = v

            rows.append(row)

    return rows


relay = get_relay(GAME_ID)
rows = parse_pitch_rows(relay)

df = pd.DataFrame(rows)

# 숫자형 변환
for col in [
    "inning", "pitch_num", "speed_kmh",
    "balls_after", "strikes_after", "outs",
    "home_score", "away_score",
    "pts_crossPlateX", "pts_crossPlateY",
    "pts_topSz", "pts_bottomSz",
    "pts_vx0", "pts_vy0", "pts_vz0",
    "pts_ax", "pts_ay", "pts_az",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 보기 좋은 순서
wanted_cols = [
    "game_id",
    "inning",
    "home_or_away",
    "pa_title",
    "pitcher_id",
    "pitcher_name",
    "batter_id",
    "batter_name",
    "pitch_num",
    "pitch_result",
    "speed_kmh",
    "pitch_type",
    "balls_after",
    "strikes_after",
    "outs",
    "base1",
    "base2",
    "base3",
    "pts_pitch_id",
    "pts_crossPlateX",
    "pts_crossPlateY",
    "pts_topSz",
    "pts_bottomSz",
    "pts_stance",
    "text",
]

existing_cols = [c for c in wanted_cols if c in df.columns]
other_cols = [c for c in df.columns if c not in existing_cols]
df = df[existing_cols + other_cols]

print(df.head(20))
print("총 투구 수:", len(df))

df.to_csv(f"kbo_pitch_{GAME_ID}.csv", index=False, encoding="utf-8-sig")

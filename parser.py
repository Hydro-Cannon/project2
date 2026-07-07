# parser.py

import time
import requests
import pandas as pd
from inning_parser import collect_relay_data_by_innings

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}


def get_relay(game_id: str):
    """
    gameId로 네이버 relay 데이터를 가져온다.
    """
    url = f"https://api-gw.sports.naver.com/schedule/games/{game_id}/relay"

    res = requests.get(url, headers=HEADERS, timeout=10)

    if res.status_code != 200:
        print("RELAY STATUS ERROR:", game_id, res.status_code)
        print(res.text[:300])
        return None

    data = res.json()
    result = data.get("result") or {}
    relay = result.get("textRelayData")

    if relay is None:
        print("relay 없음:", game_id)
        return None

    return relay


def build_player_map(relay: dict):
    """
    pcode -> 선수 이름 딕셔너리 생성.
    """
    player_map = {}

    if relay is None:
        return player_map

    for side in ["home", "away"]:
        for group in ["Lineup", "Entry"]:
            section = relay.get(f"{side}{group}") or {}

            for role in ["batter", "pitcher"]:
                for p in section.get(role, []):
                    pcode = p.get("pcode")
                    name = p.get("name")

                    if pcode and name:
                        player_map[str(pcode)] = name

    return player_map


def parse_pitch_rows(relay: dict):
    """
    relay JSON에서 pitch-level row를 추출한다.
    구종: textOptions[].stuff
    구속: textOptions[].speed
    좌표/궤적: ptsOptions
    """
    if relay is None:
        return []

    player_map = build_player_map(relay)
    rows = []

    for block_idx, block in enumerate(relay.get("textRelays", [])):
        inning = block.get("inn")
        home_or_away = block.get("homeOrAway")
        title = block.get("title")
        block_no = block.get("no")

        # ptsOptions를 pitchId 기준으로 매핑
        pts_map = {}
        for pts in block.get("ptsOptions") or []:
            pitch_id = pts.get("pitchId")
            if pitch_id:
                pts_map[pitch_id] = pts

        for opt_idx, opt in enumerate(block.get("textOptions", [])):
            # pitchNum이 없는 행은 "5번타자", "이닝 시작" 같은 설명 줄
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

                # 핵심 데이터
                "speed_kmh": opt.get("speed"),
                "pitch_type": opt.get("stuff"),

                # 투구 후 count/state
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

            # ptsOptions의 tracking data 추가
            for k, v in pts.items():
                row[f"pts_{k}"] = v

            rows.append(row)

    return rows


def parse_game(game_id: str):
    relay = get_relay(game_id)

    if relay is None:
        return pd.DataFrame()

    rows = parse_pitch_rows(relay)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def collect_pitcher_pitches(
    game_ids: list[str],
    pitcher_id: str,
    pitcher_name: str | None = None,
    sleep_sec: float = 0.5,
):
    all_pitcher_rows = []

    for game_id in sorted(set(game_ids)):
        print("game:", game_id)

        try:
            relays = collect_relay_data_by_innings(game_id, headless=True)

            if not relays:
                print("  relay 없음")
                continue

            df = parse_relays_to_dataframe(relays)

            if df.empty:
                print("  투구 row 없음")
                continue

            mask = df["pitcher_id"].astype(str) == str(pitcher_id)

            if pitcher_name is not None and "pitcher_name" in df.columns:
                mask = mask | (df["pitcher_name"] == pitcher_name)

            pitcher_df = df[mask].copy()

            if len(pitcher_df) > 0:
                pitcher_df["game_date"] = game_id[:8]
                all_pitcher_rows.append(pitcher_df)
                print("  found:", len(pitcher_df), "pitches")
            else:
                print("  해당 투수 투구 없음")

            time.sleep(sleep_sec)

        except Exception as e:
            print("relay error:", game_id, e)
            continue

    if not all_pitcher_rows:
        return pd.DataFrame()

    result_df = pd.concat(all_pitcher_rows, ignore_index=True)

    if "pts_pitch_id" in result_df.columns:
        result_df = result_df.drop_duplicates(subset=["game_id", "pts_pitch_id"])

    result_df = clean_numeric_columns(result_df)

    sort_cols = [
        "game_date",
        "game_id",
        "inning",
        "home_or_away",
        "block_no",
        "option_idx",
        "pitch_num",
    ]

    existing_sort_cols = [col for col in sort_cols if col in result_df.columns]

    result_df = result_df.sort_values(existing_sort_cols).reset_index(drop=True)

    return result_df
def clean_numeric_columns(df: pd.DataFrame):
    numeric_cols = [
        "inning",
        "pitch_num",
        "speed_kmh",
        "balls_after",
        "strikes_after",
        "outs",
        "home_score",
        "away_score",
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

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def parse_relays_to_dataframe(relays: list[dict]):
    """
    여러 회차 relay JSON을 하나의 pitch DataFrame으로 합친다.
    경기 순 -> 이닝 순 -> 타석/블록 순 -> 투구 순으로 정렬한다.
    """
    all_rows = []

    for relay in relays:
        rows = parse_pitch_rows(relay)
        all_rows.extend(rows)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # 같은 투구가 중복 수집될 수 있으므로 제거
    if "pts_pitch_id" in df.columns:
        df = df.drop_duplicates(subset=["game_id", "pts_pitch_id"])
    else:
        df = df.drop_duplicates()

    # 숫자 컬럼 변환
    df = clean_numeric_columns(df)

    # game_date가 없으면 game_id 앞 8자리로 생성
    if "game_date" not in df.columns and "game_id" in df.columns:
        df["game_date"] = df["game_id"].astype(str).str[:8]

    # 경기 순 -> 투구 순 정렬
    sort_cols = [
        "game_date",
        "game_id",
        "inning",
        "home_or_away",
        "block_no",
        "option_idx",
        "pitch_num",
    ]

    existing_sort_cols = [col for col in sort_cols if col in df.columns]

    df = df.sort_values(existing_sort_cols).reset_index(drop=True)

    return df

def save_pitcher_csv(
    df: pd.DataFrame,
    pitcher_name: str,
    pitcher_id: str,
    output_path: str | None = None,
):
    if output_path is None:
        output_path = f"kbo_pitcher_{pitcher_name}_{pitcher_id}.csv"

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return output_path

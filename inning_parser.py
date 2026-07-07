# inning_parser.py

import time
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}


def get_relay_json(game_id: str, inning: int | None = None):
    if inning is None:
        url = f"https://api-gw.sports.naver.com/schedule/games/{game_id}/relay"
    else:
        url = f"https://api-gw.sports.naver.com/schedule/games/{game_id}/relay?inning={inning}"

    res = requests.get(url, headers=HEADERS, timeout=10)

    if res.status_code != 200:
        print("relay request failed:", url, res.status_code)
        print(res.text[:200])
        return None

    data = res.json()
    result = data.get("result") or {}
    relay = result.get("textRelayData")

    if not relay:
        print("relay empty:", url)
        return None

    return relay


def get_max_inning(relay: dict):
    """
    기본 relay에서 경기의 마지막 이닝을 추정한다.
    """
    candidates = []

    for key in ["inn", "inning"]:
        value = relay.get(key)
        if value is not None:
            try:
                candidates.append(int(value))
            except Exception:
                pass

    state = relay.get("currentGameState") or {}
    for key in ["inn", "inning"]:
        value = state.get(key)
        if value is not None:
            try:
                candidates.append(int(value))
            except Exception:
                pass

    for block in relay.get("textRelays", []):
        value = block.get("inn")
        if value is not None:
            try:
                candidates.append(int(value))
            except Exception:
                pass

    if not candidates:
        return 9

    return max(candidates)


def relay_has_pitch_rows(relay: dict):
    for block in relay.get("textRelays", []):
        for opt in block.get("textOptions", []):
            if "pitchNum" in opt:
                return True
    return False


def collect_relay_data_by_innings(game_id: str, headless: bool = True, sleep_sec: float = 0.2):
    """
    gameId의 모든 이닝 relay를 직접 API로 수집한다.
    headless 인자는 기존 parser.py 호환용으로만 둔다.
    """

    relays = []

    # 1. 기본 relay로 마지막 이닝 확인
    base_relay = get_relay_json(game_id)

    if not base_relay:
        return []

    max_inning = get_max_inning(base_relay)
    print("max inning:", max_inning)

    # 기본 relay도 보관해도 되지만, 중복이 많으므로 이닝별 API를 우선 사용
    # relays.append(base_relay)

    # 2. 1회부터 마지막 회까지 직접 호출
    for inning in range(1, max_inning + 1):
        relay = get_relay_json(game_id, inning=inning)

        if not relay:
            continue

        block_innings = sorted({
            str(block.get("inn"))
            for block in relay.get("textRelays", [])
            if block.get("inn") is not None
        })

        pitch_count = 0
        for block in relay.get("textRelays", []):
            for opt in block.get("textOptions", []):
                if "pitchNum" in opt:
                    pitch_count += 1

        print(
            "relay captured:",
            f"{inning}회",
            "block innings:",
            block_innings,
            "textRelays:",
            len(relay.get("textRelays", [])),
            "pitches:",
            pitch_count,
        )

        if relay_has_pitch_rows(relay):
            relays.append(relay)

        time.sleep(sleep_sec)

    return relays

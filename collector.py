# collect_player_games.py

import re
from playwright.sync_api import sync_playwright

GAME_ID_PATTERN = re.compile(r"20\d{6}[A-Z]{4}\d{5}")


def collect_player_game_ids(player_id: str, max_expand_clicks: int = 50, headless: bool = True):
    url = (
        f"https://m.sports.naver.com/player/index"
        f"?from=nx&playerId={player_id}&category=kbo&tab=record"
    )

    game_ids = set()

    def collect_game_ids_from_text(text):
        for gid in GAME_ID_PATTERN.findall(text or ""):
            game_ids.add(gid)

    def handle_response(response):
        response_url = response.url

        if "sports.naver.com" not in response_url and "api-gw.sports.naver.com" not in response_url:
            return

        try:
            text = response.text()
        except Exception:
            return

        collect_game_ids_from_text(text)

    def click_game_by_game_tab(page):
        selectors = [
            "button:has-text('경기별 기록')",
            "a:has-text('경기별 기록')",
            "[role=tab]:has-text('경기별 기록')",
            "text=경기별 기록",
        ]

        for selector in selectors:
            try:
                loc = page.locator(selector)

                for i in range(loc.count()):
                    target = loc.nth(i)

                    if not target.is_visible(timeout=1000):
                        continue

                    target.scroll_into_view_if_needed(timeout=3000)
                    target.click(timeout=5000)
                    page.wait_for_timeout(3000)

                    collect_game_ids_from_text(page.content())
                    return True

            except Exception:
                continue

        return False

    def click_expand_buttons(page):
        for _ in range(max_expand_clicks):
            buttons = page.locator("button:has-text('펼쳐보기')")
            count = buttons.count()

            if count == 0:
                break

            clicked = False

            for i in range(count):
                try:
                    btn = buttons.nth(i)

                    if not btn.is_visible(timeout=1000):
                        continue

                    btn.scroll_into_view_if_needed(timeout=3000)
                    btn.click(timeout=5000)

                    page.wait_for_timeout(3000)
                    page.mouse.wheel(0, 1200)
                    page.wait_for_timeout(1000)

                    collect_game_ids_from_text(page.content())

                    clicked = True
                    break

                except Exception:
                    continue

            if not clicked:
                break

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)

        page = browser.new_page(
            user_agent=(
                "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                "Version/16.0 Mobile/15E148 Safari/604.1"
            ),
            viewport={"width": 390, "height": 844},
        )

        page.on("response", handle_response)

        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(5000)

        collect_game_ids_from_text(page.content())

        clicked_tab = click_game_by_game_tab(page)

        if clicked_tab:
            for _ in range(5):
                page.mouse.wheel(0, 1200)
                page.wait_for_timeout(1000)
                collect_game_ids_from_text(page.content())

            click_expand_buttons(page)

        collect_game_ids_from_text(page.content())

        browser.close()

    return sorted(game_ids)


def save_game_ids(player_id: str, game_ids: list[str]):
    filename = f"player_{player_id}_game_ids.txt"

    with open(filename, "w", encoding="utf-8") as f:
        for gid in game_ids:
            f.write(gid + "\n")

    return filename


if __name__ == "__main__":
    PLAYER_ID = "77637"

    game_ids = collect_player_game_ids(PLAYER_ID)
    filename = save_game_ids(PLAYER_ID, game_ids)

    print("Found gameId:", len(game_ids))
    for gid in game_ids:
        print(gid)

    print("SAVED:", filename)

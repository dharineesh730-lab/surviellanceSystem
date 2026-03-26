"""
Quick test to verify Telegram bot credentials and alert sending.
Run: python test_telegram.py
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "")
_raw     = os.getenv("TELEGRAM_CHAT_IDS", "") or os.getenv("TELEGRAM_CHAT_ID", "")
CHAT_IDS = [c.strip() for c in _raw.split(",") if c.strip()]
CHAT_ID  = CHAT_IDS[0] if CHAT_IDS else ""


def test_credentials():
    print("── Checking credentials ──────────────────")
    if not TOKEN:
        print("❌  TELEGRAM_BOT_TOKEN is missing in .env")
        return False
    if not CHAT_IDS:
        print("❌  TELEGRAM_CHAT_IDS is missing in .env")
        return False
    print(f"✅  Token     : {TOKEN[:10]}…{TOKEN[-5:]}")
    for cid in CHAT_IDS:
        label = "private" if not cid.startswith("-") else "group"
        print(f"✅  Chat ID   : {cid}  ({label})")
    return True


def test_bot_info():
    print("\n── Checking bot info (getMe) ─────────────")
    resp = requests.get(f"https://api.telegram.org/bot{TOKEN}/getMe", timeout=10)
    if resp.status_code == 200:
        data = resp.json()["result"]
        print(f"✅  Bot name    : {data['first_name']}")
        print(f"✅  Bot username: @{data['username']}")
        return True
    print(f"❌  getMe failed — HTTP {resp.status_code}: {resp.text[:200]}")
    return False


def check_and_clear_webhook():
    """
    If a webhook is set, getUpdates will always return empty.
    This checks for and removes any existing webhook automatically.
    """
    print("\n── Checking webhook status ───────────────")
    resp = requests.get(
        f"https://api.telegram.org/bot{TOKEN}/getWebhookInfo", timeout=10
    )
    info = resp.json().get("result", {})
    url  = info.get("url", "")

    if url:
        print(f"⚠️   Webhook is set → {url}")
        print("     Removing webhook so getUpdates can work…")
        del_resp = requests.get(
            f"https://api.telegram.org/bot{TOKEN}/deleteWebhook", timeout=10
        )
        if del_resp.json().get("result"):
            print("✅  Webhook removed.")
        else:
            print(f"❌  Could not remove webhook: {del_resp.text[:100]}")
    else:
        print("✅  No webhook set — getUpdates is free to use.")


def find_correct_chat_id():
    """Fetch recent updates to show available chat IDs."""
    print("\n── Finding Chat ID via getUpdates ────────")

    # Use limit=100 and no offset to get all recent messages
    resp = requests.get(
        f"https://api.telegram.org/bot{TOKEN}/getUpdates",
        params={"limit": 100, "allowed_updates": ["message"]},
        timeout=10,
    )
    if resp.status_code != 200:
        print(f"❌  getUpdates failed: {resp.text[:200]}")
        return

    updates = resp.json().get("result", [])
    if not updates:
        print("⚠️   Still no updates found.")
        print("     Make sure you opened @voilationdetectionbot on Telegram and tapped START.")
        return

    seen = {}
    for u in updates:
        msg  = u.get("message", {})
        chat = msg.get("chat", {})
        if chat:
            cid   = chat.get("id")
            ctype = chat.get("type")
            name  = (chat.get("title")
                     or chat.get("username")
                     or f"{chat.get('first_name','')} {chat.get('last_name','')}".strip())
            seen[cid] = (ctype, name)

    print(f"  Found {len(seen)} chat(s):\n")
    for cid, (ctype, name) in seen.items():
        match = " ← matches your .env" if str(cid) == str(CHAT_ID) else ""
        print(f"  Chat ID : {cid}{match}")
        print(f"  Type    : {ctype}")
        print(f"  Name    : {name}")
        print()


def test_send_message():
    print("\n── Sending test message ──────────────────")
    resp = requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        data={
            "chat_id": CHAT_ID,
            "text":    "✅ Action Recognition system — Telegram alert is working!",
        },
        timeout=10,
    )
    if resp.status_code == 200:
        print("✅  Message sent successfully!")
        return True
    print(f"❌  sendMessage failed — HTTP {resp.status_code}: {resp.text[:200]}")
    return False


def test_send_video():
    """Test sending an actual video clip from tmp/ folder."""
    print("\n── Sending test video ────────────────────")
    import glob
    clips = sorted(glob.glob("tmp/*.mp4"))
    if not clips:
        print("⚠️   No video clips found in tmp/ — run main.py first to generate one.")
        return

    clip_path = clips[-1]   # use the most recent clip
    print(f"     Sending: {clip_path}")
    try:
        with open(clip_path, "rb") as video:
            resp = requests.post(
                f"https://api.telegram.org/bot{TOKEN}/sendVideo",
                data={
                    "chat_id":            CHAT_ID,
                    "caption":            "⚠️ Fight Detected — Security Alert (test)",
                    "supports_streaming": "true",
                },
                files={"video": video},
                timeout=60,
            )
        if resp.status_code == 200:
            print("✅  Video sent successfully!")
        else:
            print(f"❌  sendVideo failed — HTTP {resp.status_code}: {resp.text[:200]}")
    except requests.RequestException as exc:
        print(f"❌  Request failed: {exc}")


if __name__ == "__main__":
    print("═══════════════════════════════════════════")
    print("   Telegram Alert — Connection Test")
    print("═══════════════════════════════════════════")

    if not test_credentials():
        raise SystemExit("Fix your .env file and try again.")

    if not test_bot_info():
        raise SystemExit("Bot token is invalid. Check TELEGRAM_BOT_TOKEN.")

    # Clear any webhook that may be blocking getUpdates
    check_and_clear_webhook()

    if not test_send_message():
        find_correct_chat_id()
        print("\n💡  Steps to fix:")
        print("    1. Open Telegram → search @voilationdetectionbot → tap START")
        print("    2. Run this test again")
        print("    3. Copy the Chat ID shown above into TELEGRAM_CHAT_IDS in .env")
        raise SystemExit

    # Also test video sending if clips exist
    test_send_video()

    print("\n✅  All checks passed — Telegram is ready!")

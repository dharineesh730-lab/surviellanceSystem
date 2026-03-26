"""
Telegram Alert Module
---------------------
Sends a fight-detection video clip to one or more Telegram chats.

Telegram requires H.264 (libx264) for inline video playback.
This module automatically re-encodes the clip using ffmpeg before sending.

Configuration (set in your .env file):
    TELEGRAM_BOT_TOKEN  — your bot token from @BotFather
    TELEGRAM_CHAT_IDS   — comma-separated list of chat IDs to notify
                          e.g.  712345678,-5140879081
                          • Positive number  = private person
                          • Negative number  = group chat
"""

import os
import logging
import subprocess
import tempfile

import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

TOKEN_KEY    = os.getenv("TELEGRAM_BOT_TOKEN", "")
_raw_ids     = os.getenv("TELEGRAM_CHAT_IDS", "") or os.getenv("TELEGRAM_CHAT_ID", "")
CHAT_IDS     = [cid.strip() for cid in _raw_ids.split(",") if cid.strip()]
# ALERT_FORMAT: "video" (H.264 mp4) or "gif" (animated GIF)
ALERT_FORMAT = os.getenv("ALERT_FORMAT", "gif").lower()


def _convert_to_gif(input_path: str) -> str:
    """
    Convert video to an animated GIF using ffmpeg.
    Telegram plays GIFs as silent looping animations — no codec issues.

    Settings:
        fps=12       — smooth enough, keeps file size down
        scale=480    — 480px wide (good quality on mobile)
        loop=0       — loop forever in Telegram
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    tmp.close()
    output_path = tmp.name

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", input_path,
                "-vf", "fps=12,scale=480:-1:flags=lanczos",
                "-loop", "0",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
        log.info(f"Converted to GIF → {output_path}")
        return output_path

    except FileNotFoundError:
        log.warning("ffmpeg not found — falling back to original video.")
        os.unlink(output_path)
        return input_path

    except subprocess.CalledProcessError as exc:
        log.warning(f"GIF conversion failed: {exc.stderr.decode()[:200]}")
        os.unlink(output_path)
        return input_path


def _convert_to_h264(input_path: str) -> str:
    """
    Re-encode a video to H.264 + AAC using ffmpeg so Telegram
    plays it inline instead of showing a single frozen frame.

    Returns the path to the converted file (a temp file).
    Falls back to the original path if ffmpeg is not available.
    """
    tmp = tempfile.NamedTemporaryFile(suffix="_h264.mp4", delete=False)
    tmp.close()
    output_path = tmp.name

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",                    # overwrite output if exists
                "-i", input_path,        # input file
                "-vcodec", "libx264",    # H.264 video — required by Telegram
                "-preset", "fast",       # fast encoding (lower CPU cost)
                "-crf", "23",            # quality (18=best, 28=worst, 23=default)
                "-pix_fmt", "yuv420p",   # pixel format Telegram needs
                "-movflags", "+faststart",  # stream-friendly MP4 header
                "-an",                   # no audio track (alert clips have none)
                output_path,
            ],
            check=True,
            capture_output=True,
        )
        log.info(f"Re-encoded to H.264 → {output_path}")
        return output_path

    except FileNotFoundError:
        log.warning("ffmpeg not found — sending original clip (may not play inline).")
        os.unlink(output_path)
        return input_path

    except subprocess.CalledProcessError as exc:
        log.warning(f"ffmpeg re-encode failed: {exc.stderr.decode()[:200]}")
        os.unlink(output_path)
        return input_path


def send_image(file_name: str) -> int:
    """
    Re-encode and send a fight clip to every configured Telegram chat.

    Parameters
    ----------
    file_name : str  — file inside ``tmp/`` to send.

    Returns
    -------
    int  — 200 on success, 400 on missing file/credentials,
           or the first non-200 HTTP status encountered.
    """
    if not TOKEN_KEY or not CHAT_IDS:
        log.warning(
            "Telegram credentials not configured. "
            "Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_IDS to your .env file."
        )
        return 400

    original_path = os.path.join("tmp", file_name)
    if not os.path.isfile(original_path):
        log.warning(f"Alert file not found: {original_path}")
        return 400

    # Convert to the configured format before sending
    if ALERT_FORMAT == "gif":
        send_path   = _convert_to_gif(original_path)
        api_method  = "sendAnimation"
        file_key    = "animation"
    else:
        send_path   = _convert_to_h264(original_path)
        api_method  = "sendVideo"
        file_key    = "video"

    is_temp  = send_path != original_path
    url      = f"https://api.telegram.org/bot{TOKEN_KEY}/{api_method}"
    caption  = "⚠️ Fight Detected — Security Alert"
    last_status = 200

    try:
        for chat_id in CHAT_IDS:
            try:
                with open(send_path, "rb") as media:
                    resp = requests.post(
                        url,
                        data={
                            "chat_id": chat_id,
                            "caption": caption,
                        },
                        files={file_key: media},
                        timeout=60,
                    )
                if resp.status_code == 200:
                    log.info(f"{ALERT_FORMAT.upper()} alert sent → chat {chat_id}")
                else:
                    log.warning(
                        f"Video alert to chat {chat_id} failed — "
                        f"HTTP {resp.status_code}: {resp.text[:120]}"
                    )
                    last_status = resp.status_code

            except requests.RequestException as exc:
                log.error(f"Telegram request to chat {chat_id} failed: {exc}")
                last_status = 500
    finally:
        # Clean up the temp H.264 file
        if is_temp and os.path.exists(send_path):
            os.unlink(send_path)

    return last_status

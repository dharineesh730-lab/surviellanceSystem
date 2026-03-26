# Action Recognition — SlowFast + YOLOv8 + DeepSort

Real-time video action recognition that detects people, tracks them across
frames, and classifies what they are doing using the SlowFast R50 network
trained on the AVA dataset (80 action classes).
When a **fight** is detected a short clip is automatically sent as a
Telegram alert.

---

## Pipeline

```
Video frame
   │
   ▼
YOLOv8 (pretrained)          ← person detection
   │  boxes + confidence scores
   ▼
DeepSort                      ← multi-object tracking
   │  persistent track IDs
   ▼
SlowFast R50  (every 50 frames)  ← action classification
   │  top-1 AVA action label per person
   ▼
Annotated output video  +  optional Telegram fight alert
```

---

## Project Structure

```
ActionRecognitionSlowfast/
├── main.py                        # entry point — runs the full pipeline
├── requirements.txt
├── .env.example                   # copy to .env and fill credentials
│
├── models/
│   └── ckpt.t7                    # DeepSort ReID checkpoint (you provide this)
│
├── selfutils/
│   ├── __init__.py
│   ├── utils.py                   # MyVideoCapture + save_video
│   ├── telegram_api.py            # fight-detection Telegram alert
│   ├── ava_action_list.pbtxt      # AVA action label map (80 classes)
│   └── visualization.py           # (legacy) Detectron2-based visualiser
│
└── deep_sort/
    └── deep_sort/
        ├── deep_sort.py           # tracker interface
        ├── sort/                  # Kalman filter, Hungarian algorithm, IoU
        └── deep/                  # ReID feature extractor + model
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU strongly recommended.**
> Make sure your `torch` version matches your CUDA version.

### 2. Configure Telegram alerts (optional)

```bash
cp .env.example .env
# then edit .env and add your bot token and chat ID
```

### 3. Place the DeepSort ReID model

Put your `ckpt.t7` file in the `models/` folder:

```
models/ckpt.t7
```

### 4. Run

```bash
# On a video file
python main.py --input fight_0001.mp4 --output result.mp4 --device cuda

# On a webcam (index 0)
python main.py --input 0 --output result.mp4 --show

# CPU-only (slower)
python main.py --input fight_0001.mp4 --output result.mp4 --device cpu
```

---

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | `fight_0001.mp4` | Video file path or camera index |
| `--output` | `output.mp4` | Annotated output video path |
| `--yolo-model` | `yolov8m.pt` | YOLOv8 weights (auto-downloaded if absent) |
| `--reid-model` | `models/ckpt.t7` | DeepSort ReID checkpoint |
| `--imsize` | `640` | Inference resolution (pixels) |
| `--conf` | `0.4` | Detection confidence threshold |
| `--iou` | `0.45` | NMS IoU threshold |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--show` | flag | Display live annotated video |

---

## Models Used

| Model | Purpose | Source |
|---|---|---|
| **YOLOv8m** | Person detection | `ultralytics` (auto-downloaded) |
| **SlowFast R50** | Action recognition (80 AVA classes) | PyTorch Hub (auto-downloaded) |
| **DeepSort ReID** | Person re-identification across frames | Custom checkpoint (`models/ckpt.t7`) |

YOLOv8m is chosen for the best balance of speed and accuracy on person
detection.  You can switch to `yolov8l.pt` for higher accuracy or
`yolov8n.pt` for faster inference.

---

## Fight Detection & Alerts

- The system watches for AVA action class **64** ("fight person").
- When detected, a short clip (±25 frames around the event) is saved to
  `tmp/` and sent to the configured Telegram chat.
- A **30-second cooldown** prevents duplicate alerts for the same event.
- Alerts run in a background thread so they never stall the inference loop.

---

## .env Variables

| Variable | Description |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Bot token from [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | Target chat ID (prefix with `-` for groups) |

If credentials are missing the alert is silently skipped and a warning is
logged — the rest of the pipeline continues normally.

---

## Notes

- Action recognition runs every **50 frames** (~2 seconds at 25 fps).
  Between clips, each person shows its most recent action label.
- The DeepSort tracker assigns persistent IDs even when persons leave and
  re-enter the frame (up to `MAX_AGE = 70` frames gap).
- All intermediate fight clips are saved in `tmp/` for review.

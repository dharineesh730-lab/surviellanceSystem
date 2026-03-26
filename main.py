"""
Action Recognition with SlowFast + YOLOv8 + DeepSort
=====================================================

Pipeline
--------
  1. YOLOv8  – detects persons in every frame (pretrained on COCO)
  2. DeepSort – assigns a persistent track ID to each person
  3. SlowFast – classifies actions every CLIP_LEN frames (80 AVA classes)
  4. Telegram – sends a fight-clip alert when class FIGHT_CLASS_IDX fires

Usage
-----
  python main.py --input fight_0001.mp4 --output result.mp4 --device cuda

  # Webcam:
  python main.py --input 0 --output result.mp4 --show
"""

import os
import cv2
import time
import queue
import logging
import argparse
import threading
import warnings
import colorsys

import numpy as np
import torch

from dotenv import load_dotenv
from ultralytics import YOLO
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,
)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection

from deep_sort.deep_sort import DeepSort
from selfutils import MyVideoCapture, save_video, send_image

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

# ──────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────
CLIP_LEN             = 50    # frames buffered before each SlowFast call
FIGHT_CLASS_IDX      = 63    # 0-indexed AVA action class for "fight person"
FIGHT_COOLDOWN_SECS  = 0.0   # 0 = send an alert for every detected fight clip
                             # set > 0 (e.g. 30.0) to suppress repeated alerts
ALERT_FRAME_BUF      = 75    # frames around the event (±75 @ 25fps = 6 sec clip)
TOP_K_ACTIONS        = 5     # top-k action predictions retrieved per person
AVA_LABEL_MAP        = os.path.join("selfutils", "ava_action_list.pbtxt")


# ──────────────────────────────────────────────────────────
# Colour map (80 visually distinct BGR colours)
# ──────────────────────────────────────────────────────────
def _build_color_map(n: int = 80):
    colors = []
    for i in range(n):
        h = i / n
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))
    return colors

COLOR_MAP = _build_color_map(80)


# ──────────────────────────────────────────────────────────
# Video Transform (SlowFast pre-processing)
# ──────────────────────────────────────────────────────────
def ava_inference_transform(
    clip,
    boxes,
    num_frames: int = 32,
    crop_size: int = 640,
    data_mean=(0.45, 0.45, 0.45),
    data_std=(0.225, 0.225, 0.225),
    slow_fast_alpha: int = 4,
):
    """
    Prepare a raw video clip and person bounding boxes for SlowFast inference.

    Parameters
    ----------
    clip         : Tensor  (C, T, H, W) — raw uint8 video frames
    boxes        : ndarray (N, 4)       — xyxy boxes from the tracker
    num_frames   : int    — fast-pathway frame count (slow = num_frames // alpha)
    crop_size    : int    — shorter spatial side after scaling
    data_mean/std: tuple  — ImageNet-style normalisation values
    slow_fast_alpha: int  — temporal stride ratio (fast / slow)

    Returns
    -------
    inputs   : list[Tensor]  — [slow_pathway, fast_pathway] ready for the model
    inp_boxes: Tensor (N,4)  — scaled & clipped boxes aligned with the crop
    roi_boxes: ndarray       — original (unscaled) boxes, kept for reference
    """
    boxes = np.array(boxes)
    roi_boxes = boxes.copy()

    # Temporal sub-sampling
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float() / 255.0

    # Clip boxes to image boundaries before scaling
    h, w = clip.shape[2], clip.shape[3]
    boxes = clip_boxes_to_image(boxes, h, w)

    # Scale shortest side and adjust boxes accordingly
    clip, boxes = short_side_scale_with_boxes(clip, size=crop_size, boxes=boxes)

    # Normalise
    clip = normalize(
        clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),
    )

    # Clip boxes to the new (scaled) frame size
    boxes = clip_boxes_to_image(boxes, clip.shape[2], clip.shape[3])

    # Build slow / fast pathways
    fast_pathway = clip
    slow_indices = torch.linspace(
        0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha
    ).long()
    slow_pathway = torch.index_select(clip, 1, slow_indices)

    return [slow_pathway, fast_pathway], torch.from_numpy(boxes), roi_boxes


# ──────────────────────────────────────────────────────────
# Person Detection  (YOLOv8)
# ──────────────────────────────────────────────────────────
def detect_persons(model, frame, conf: float, iou: float, imsize: int):
    """
    Run YOLOv8 on a single BGR frame and return person detections.

    Returns
    -------
    boxes_xyxy  : Tensor (N, 4) — absolute pixel corners
    boxes_xywh  : Tensor (N, 4) — center_x, center_y, width, height
    confidences : Tensor (N,)   — detection scores
    classes     : Tensor (N,)   — class IDs (0 = person)
    """
    results = model(frame, conf=conf, iou=iou, classes=[0],
                    imgsz=imsize, verbose=False)
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        empty = torch.zeros((0, 4))
        return empty, empty, torch.zeros(0), torch.zeros(0)

    return (
        result.boxes.xyxy.cpu(),
        result.boxes.xywh.cpu(),
        result.boxes.conf.cpu(),
        result.boxes.cls.cpu(),
    )


# ──────────────────────────────────────────────────────────
# Action Recognition  (SlowFast R50)
# ──────────────────────────────────────────────────────────
def run_slowfast(video_model, clip, person_boxes, ava_labels, device, crop_size):
    """
    Run SlowFast on a buffered clip to classify actions for each tracked person.

    Parameters
    ----------
    video_model  : nn.Module  — SlowFast R50 loaded from PyTorch Hub
    clip         : Tensor     — (C, T, H, W) video clip
    person_boxes : ndarray    — (N, 4) xyxy boxes (from tracker)
    ava_labels   : dict       — {int id → str label}
    device       : torch.device
    crop_size    : int

    Returns
    -------
    action_labels  : list[str]  — top-1 action name per detected person
    fight_detected : bool       — True if FIGHT_CLASS_IDX appears in any top-k
    """
    inputs, inp_boxes, _ = ava_inference_transform(
        clip, person_boxes, crop_size=crop_size
    )

    # Prepend a batch-index column (all zeros → single batch)
    inp_boxes = torch.cat(
        [torch.zeros(inp_boxes.shape[0], 1), inp_boxes], dim=1
    )

    # Move to device
    inputs = [inp.unsqueeze(0).to(device) for inp in inputs]

    with torch.no_grad():
        preds = video_model(inputs, inp_boxes.to(device)).cpu()

    action_labels = []
    fight_detected = False

    for logits in preds:
        # Sort by score descending
        top_indices = np.argsort(logits.numpy())[::-1][:TOP_K_ACTIONS]

        if FIGHT_CLASS_IDX in top_indices:
            fight_detected = True
            primary_idx = FIGHT_CLASS_IDX
        else:
            primary_idx = int(top_indices[0])

        # AVA label map is 1-indexed; model outputs are 0-indexed
        label = ava_labels.get(primary_idx + 1, "unknown")
        action_labels.append(label)

    return action_labels, fight_detected


# ──────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────
def draw_boxes(frame, tracked_outputs, id_to_ava_labels: dict):
    """
    Overlay bounding boxes, track IDs, and action labels on *frame* (in-place).

    Parameters
    ----------
    frame           : ndarray (H, W, 3)  — BGR image
    tracked_outputs : ndarray (N, 8)     — [x1,y1,x2,y2, cls, tid, vx, vy]
    id_to_ava_labels: dict               — track_id → action label string
    """
    if tracked_outputs is None or len(tracked_outputs) == 0:
        return frame

    for row in tracked_outputs:
        x1, y1, x2, y2, cls, tid, vx, vy = row
        c1 = (int(x1), int(y1))
        c2 = (int(x2), int(y2))
        color = COLOR_MAP[int(cls) % len(COLOR_MAP)]

        action = id_to_ava_labels.get(tid, "tracking…")
        label  = f"ID:{int(tid)}  {action}"

        # Draw bounding box
        cv2.rectangle(frame, c1, c2, color, thickness=2, lineType=cv2.LINE_AA)

        # Draw label background + text
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        label_y = max(c1[1] - 6, th + 4)
        cv2.rectangle(
            frame,
            (c1[0], label_y - th - baseline - 2),
            (c1[0] + tw, label_y),
            color,
            -1,
        )
        cv2.putText(
            frame, label,
            (c1[0], label_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

    return frame


# ──────────────────────────────────────────────────────────
# Fight Alert  (runs in background thread)
# ──────────────────────────────────────────────────────────
def handle_fight_alert(cap: MyVideoCapture, frame_idx: int):
    """
    Extract a short clip around the detected fight and send it via Telegram.
    Runs in a daemon thread so it never blocks the main inference loop.
    """
    os.makedirs("tmp", exist_ok=True)
    file_name = f"fight_{frame_idx}.mp4"
    frames = cap.get_frames_around_index(
        index=frame_idx, frame_buffer=ALERT_FRAME_BUF
    )
    if frames:
        save_video(frame_list=frames, dst=os.path.join("tmp", file_name))
        status = send_image(file_name=file_name)
        if status == 200:
            log.info("Fight alert sent successfully via Telegram.")
        else:
            log.warning(f"Fight alert failed — Telegram returned HTTP {status}.")
    else:
        log.warning("No frames available for fight alert clip.")


# ──────────────────────────────────────────────────────────
# SlowFast Queue Worker
# ──────────────────────────────────────────────────────────
def slowfast_queue_worker(
    model, ava_labels, device, imsize,
    id_to_ava_labels, labels_lock,
    fight_state, fight_lock,
    cap_ref, clip_queue,
    cooldown_frames, alert_threads, alert_threads_lock,
):
    """
    Long-running background thread that drains a clip queue one at a time.

    Why a queue instead of spawning a new thread per clip:
    - Prevents clips from being skipped when a previous one is still running
    - Guarantees every clip is processed in order
    - Main loop never blocks — it just enqueues and moves on
    - Sending None into the queue is the shutdown signal (sentinel pattern)

    Queue item format: (clip, tracked_snap, frame_idx, clip_num)

    Cooldown is VIDEO-frame-based (not wall-clock time) so it behaves
    consistently regardless of how fast or slow the hardware processes clips.
    """
    while True:
        item = clip_queue.get()
        if item is None:          # sentinel — time to stop
            clip_queue.task_done()
            break

        clip, tracked_snap, frame_idx, clip_num = item
        log.info(
            f"Clip #{clip_num} — running SlowFast "
            f"({len(tracked_snap)} persons, frame {frame_idx})"
        )
        try:
            person_boxes = tracked_snap[:, :4]
            action_labels, fight_detected = run_slowfast(
                model, clip, person_boxes, ava_labels, device, imsize
            )
            with labels_lock:
                for row, label in zip(tracked_snap, action_labels):
                    id_to_ava_labels[row[5]] = label

            if fight_detected:
                with fight_lock:
                    frames_since_last = frame_idx - fight_state["last_alert_frame"]
                    if frames_since_last >= cooldown_frames:
                        fight_state["last_alert_frame"] = frame_idx
                        do_alert = True
                    else:
                        remaining = cooldown_frames - frames_since_last
                        log.info(
                            f"Fight suppressed (cooldown: {remaining} more "
                            f"video-frames needed before next alert)"
                        )
                        do_alert = False

                if do_alert:
                    log.warning(
                        f"⚠  FIGHT DETECTED at frame {frame_idx} — "
                        f"sending Telegram alert!"
                    )
                    # Non-daemon so we can join() it before releasing the cap
                    t = threading.Thread(
                        target=handle_fight_alert,
                        args=(cap_ref, frame_idx),
                        daemon=False,
                    )
                    with alert_threads_lock:
                        alert_threads.append(t)
                    t.start()

        except Exception as exc:
            log.error(f"SlowFast clip #{clip_num} failed: {exc}")
        finally:
            clip_queue.task_done()


# ──────────────────────────────────────────────────────────
# Main Inference Loop
# ──────────────────────────────────────────────────────────
def main(config):
    # ── Device selection ─────────────────────────────────
    # YOLOv8 (2D) — use the best available accelerator
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # SlowFast (3D) — MPS does not support Conv3D yet, fall back to CPU.
    # On a CUDA machine both models share the GPU.
    if torch.cuda.is_available():
        slowfast_device = torch.device("cuda")
    else:
        slowfast_device = torch.device("cpu")

    log.info(f"YOLOv8 device  : {device}")
    log.info(f"SlowFast device: {slowfast_device}")

    # ── Load models ──────────────────────────────────────
    log.info(f"Loading YOLOv8 model: {config.yolo_model}")
    yolo = YOLO(config.yolo_model)

    log.info("Loading SlowFast R50 (pretrained on AVA)…")
    slowfast = slowfast_r50_detection(pretrained=True).eval().to(slowfast_device)

    log.info(f"Loading DeepSort tracker: {config.reid_model}")
    tracker = DeepSort(config.reid_model)

    ava_labels, _ = AvaLabeledVideoFramePaths.read_label_map(AVA_LABEL_MAP)
    log.info(f"AVA label map loaded — {len(ava_labels)} action classes.")

    # ── Video I/O ────────────────────────────────────────
    probe  = cv2.VideoCapture(config.input)
    width  = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = probe.get(cv2.CAP_PROP_FPS) or 25.0
    probe.release()

    os.makedirs(os.path.dirname(os.path.abspath(config.output)), exist_ok=True)
    out_writer = cv2.VideoWriter(
        config.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    log.info(f"Output: {config.output}  ({width}×{height} @ {fps:.1f} fps)")

    cap = MyVideoCapture(config.input)

    # ── Shared state (accessed by main thread + SlowFast worker) ─────
    id_to_ava_labels = {}
    labels_lock      = threading.Lock()

    # Cooldown in video frames (not wall time) so it's independent of speed.
    # 0.0 s → every fight clip sends an alert  (no suppression)
    # 30.0 s → at most one alert per 30 video-seconds (e.g. 750 frames @ 25 fps)
    fight_cooldown_frames = int(fps * config.fight_cooldown)
    # Initialise to -(cooldown+1) so the very first detection always fires,
    # even when cooldown_frames == 0 (avoids 0 - 0 = 0 >= 0 being the boundary).
    fight_state      = {"last_alert_frame": -(fight_cooldown_frames + 1)}
    fight_lock       = threading.Lock()

    # Alert threads tracked for safe joining before cap release
    alert_threads      = []
    alert_threads_lock = threading.Lock()

    # ── Clip queue — main loop enqueues, worker drains sequentially ──
    clip_queue   = queue.Queue()
    worker       = threading.Thread(
        target=slowfast_queue_worker,
        args=(
            slowfast, ava_labels, slowfast_device, config.imsize,
            id_to_ava_labels, labels_lock,
            fight_state, fight_lock,
            cap, clip_queue,
            fight_cooldown_frames, alert_threads, alert_threads_lock,
        ),
        daemon=False,   # NOT daemon — we join() it explicitly at the end
    )
    worker.start()

    # ── Frame store for deferred video writing ────────────────────────
    # Frames are stored during the fast main loop and written AFTER
    # all SlowFast clips complete so that action labels appear correctly.
    frame_store  = []   # list of (frame ndarray, tracked ndarray)

    last_tracked = np.zeros((0, 8), dtype=np.float32)
    clip_count   = 0
    t_start      = time.time()

    log.info(f"Detection every {config.detect_every} frame(s). "
             f"SlowFast every {CLIP_LEN} frames.")
    log.info("Pass 1/2 — detection + tracking + queuing SlowFast clips…")

    while not cap.end:
        ret, frame = cap.read()
        if not ret:
            continue

        # ── 1. Person detection (YOLOv8) — skip non-key frames ──────
        if cap.idx % config.detect_every == 0:
            boxes_xyxy, boxes_xywh, confs, classes = detect_persons(
                yolo, frame, config.conf, config.iou, config.imsize
            )
            tracked = np.zeros((0, 8), dtype=np.float32)
            if len(boxes_xywh) > 0:
                result = tracker.update(
                    boxes_xywh,
                    confs.unsqueeze(1),
                    classes.tolist(),
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                if isinstance(result, np.ndarray) and result.ndim == 2:
                    tracked = result.astype(np.float32)
            last_tracked = tracked
        else:
            tracked = last_tracked   # reuse Kalman-predicted positions

        # ── 2. Buffer frame for deferred writing ─────────────────────
        frame_store.append((frame.copy(), tracked.copy()))

        # ── 3. Enqueue clip for SlowFast (every CLIP_LEN frames) ─────
        if len(cap.stack) == CLIP_LEN:
            clip_count += 1
            clip = cap.get_video_clip()   # clears the buffer
            if len(tracked) > 0:
                clip_queue.put((clip, tracked.copy(), cap.idx, clip_count))
                log.info(
                    f"Clip #{clip_count} queued "
                    f"(frame {cap.idx}, {len(tracked)} persons)"
                )

    # ── Pass 1 done — signal worker to stop and wait for it ──────────
    t_pass1 = time.time() - t_start
    log.info(
        f"Pass 1 done in {t_pass1:.1f}s "
        f"({clip_count} clip(s) queued). "
        f"Waiting for SlowFast…"
    )
    clip_queue.put(None)   # sentinel
    worker.join()          # blocks until ALL clips are processed
    log.info("SlowFast complete — all action labels ready.")

    # Wait for any in-flight Telegram alert threads to finish sending
    # BEFORE releasing the cap (they may still be reading from the video file).
    with alert_threads_lock:
        pending = list(alert_threads)
    if pending:
        log.info(f"Waiting for {len(pending)} Telegram alert(s) to finish…")
        for t in pending:
            t.join()
        log.info("All Telegram alerts sent.")

    # ── Pass 2 — write output video with correct labels ───────────────
    log.info(f"Pass 2/2 — writing {len(frame_store)} annotated frames…")
    with labels_lock:
        final_labels = dict(id_to_ava_labels)

    for i, (frame, tracked) in enumerate(frame_store):
        annotated = draw_boxes(frame, tracked, final_labels)
        out_writer.write(annotated)
        if config.show:
            cv2.imshow("Action Recognition", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                log.info("Q pressed — stopping early.")
                break

    # ── Wrap up ───────────────────────────────────────────────────────
    elapsed    = time.time() - t_start
    video_secs = cap.idx / fps
    log.info(
        f"Done.  Frames: {cap.idx}  |  "
        f"Video: {video_secs:.1f}s  |  "
        f"Total wall time: {elapsed:.1f}s"
    )

    cap.release()
    out_writer.release()
    if config.show:
        cv2.destroyAllWindows()

    log.info(f"Output saved → {config.output}")


# ──────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SlowFast Action Recognition · YOLOv8 + DeepSort",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # I/O
    parser.add_argument("--input",  type=str, default="fight_0001.mp4",
                        help="Input video path or camera index (e.g. 0)")
    parser.add_argument("--output", type=str, default="output.mp4",
                        help="Output annotated video path")
    # Models
    parser.add_argument("--yolo-model", dest="yolo_model",
                        type=str, default="yolov8m.pt",
                        help="YOLOv8 weights file (auto-downloaded if absent)")
    parser.add_argument("--reid-model", dest="reid_model",
                        type=str, default="models/ckpt.t7",
                        help="DeepSort ReID checkpoint (.t7)")
    # Detection settings
    parser.add_argument("--imsize", type=int,   default=640,
                        help="Inference image size (pixels)")
    parser.add_argument("--conf",   type=float, default=0.4,
                        help="Detection confidence threshold")
    parser.add_argument("--iou",    type=float, default=0.45,
                        help="NMS IoU threshold")
    # Speed
    parser.add_argument("--detect-every", dest="detect_every",
                        type=int, default=2,
                        help="Run YOLOv8 every N frames (1=every frame, "
                             "2=every other frame, 3=every 3rd, …). "
                             "Higher = faster but slightly less accurate.")
    # Alerts
    parser.add_argument("--fight-cooldown", dest="fight_cooldown",
                        type=float, default=FIGHT_COOLDOWN_SECS,
                        help="Minimum VIDEO seconds between Telegram alerts. "
                             "0 = send a GIF for every clip that detects a fight "
                             "(default). Use e.g. 30 to suppress repeated alerts "
                             "for the same incident.")
    # Compute
    parser.add_argument("--device", type=str, default="cuda",
                        help="Compute device: cuda | cpu")
    # Display
    parser.add_argument("--show", action="store_true",
                        help="Display annotated video while processing")

    config = parser.parse_args()

    # ── Validate ─────────────────────────────────────────
    if not str(config.input).isdigit() and not os.path.isfile(config.input):
        parser.error(f"Input file not found: {config.input}")
    if not os.path.isfile(config.reid_model):
        parser.error(
            f"ReID model not found: {config.reid_model}\n"
            "Place your ckpt.t7 inside the models/ folder."
        )
    if not os.path.isfile(AVA_LABEL_MAP):
        parser.error(f"AVA label map not found: {AVA_LABEL_MAP}")

    if str(config.input).isdigit():
        config.input = int(config.input)

    os.makedirs("tmp", exist_ok=True)

    log.info(f"Config: {vars(config)}")
    main(config)

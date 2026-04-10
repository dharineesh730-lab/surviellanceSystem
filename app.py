"""
Web UI for Action Recognition — SlowFast + YOLOv8 + DeepSort
=============================================================
Flask application that wraps the existing CLI pipeline behind
a drag-and-drop video upload interface.

Run:
    python app.py            # starts on http://localhost:5000
    python app.py --port 8080
"""

import os
import uuid
import time
import queue
import logging
import argparse
import threading
import warnings
import colorsys
import glob

import cv2
import numpy as np
import torch
from flask import (
    Flask, render_template, request, jsonify,
    send_from_directory, Response,
)

from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB upload limit

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("tmp", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Job state — tracks every processing job by ID
# ──────────────────────────────────────────────────────────
jobs = {}  # job_id → { status, progress, phase, logs, output_file, ... }
jobs_lock = threading.Lock()

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "wmv", "flv", "webm"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def update_job(job_id, **kwargs):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(kwargs)


def append_log(job_id, message):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["logs"].append(message)


# ──────────────────────────────────────────────────────────
# Import pipeline components from existing main.py
# ──────────────────────────────────────────────────────────
from main import (
    ava_inference_transform,
    detect_persons,
    run_slowfast,
    draw_boxes,
    handle_fight_alert,
    CLIP_LEN,
    FIGHT_CLASS_IDX,
    FIGHT_COOLDOWN_SECS,
    ALERT_FRAME_BUF,
    TOP_K_ACTIONS,
    AVA_LABEL_MAP,
    COLOR_MAP,
)
from selfutils import MyVideoCapture, save_video, send_image
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths


# ──────────────────────────────────────────────────────────
# Model cache (load once, reuse across jobs)
# ──────────────────────────────────────────────────────────
_model_cache = {}
_model_lock = threading.Lock()


def get_models(yolo_model_path, reid_model_path, imsize, preferred_device="cpu"):
    """Load models lazily and cache them for reuse."""
    with _model_lock:
        cache_key = (yolo_model_path, reid_model_path, preferred_device)
        if cache_key in _model_cache:
            return _model_cache[cache_key]

    from ultralytics import YOLO
    from pytorchvideo.models.hub import slowfast_r50_detection
    from deep_sort.deep_sort import DeepSort

    if preferred_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        slowfast_device = torch.device("cuda")
    elif preferred_device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        # SlowFast uses Conv3D which MPS doesn't support — must stay on CPU
        slowfast_device = torch.device("cpu")
    elif preferred_device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            slowfast_device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            slowfast_device = torch.device("cpu")
        else:
            device = torch.device("cpu")
            slowfast_device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        slowfast_device = torch.device("cpu")

    yolo = YOLO(yolo_model_path)
    slowfast = slowfast_r50_detection(pretrained=True).eval().to(slowfast_device)
    tracker = DeepSort(reid_model_path)
    ava_labels, _ = AvaLabeledVideoFramePaths.read_label_map(AVA_LABEL_MAP)

    models = {
        "yolo": yolo,
        "slowfast": slowfast,
        "tracker": tracker,
        "ava_labels": ava_labels,
        "device": device,
        "slowfast_device": slowfast_device,
    }

    with _model_lock:
        _model_cache[cache_key] = models

    return models


# ──────────────────────────────────────────────────────────
# Background processing worker
# ──────────────────────────────────────────────────────────
def process_video(job_id, input_path, params):
    """Run the full detection → tracking → action recognition pipeline."""
    try:
        update_job(job_id, status="loading", phase="Loading models...")
        append_log(job_id, "Loading models...")

        models = get_models(
            params["yolo_model"],
            params["reid_model"],
            params["imsize"],
            preferred_device=params.get("device", "cpu"),
        )
        yolo = models["yolo"]
        slowfast = models["slowfast"]
        ava_labels = models["ava_labels"]
        device = models["device"]
        slowfast_device = models["slowfast_device"]

        # DeepSort must be re-instantiated per job (it's stateful)
        from deep_sort.deep_sort import DeepSort
        tracker = DeepSort(params["reid_model"])

        append_log(job_id, f"YOLOv8 device: {device}")
        append_log(job_id, f"SlowFast device: {slowfast_device}")
        append_log(job_id, "Models loaded successfully.")

        # Video probe
        probe = cv2.VideoCapture(input_path)
        width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = probe.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
        probe.release()

        output_filename = f"{job_id}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # Try H.264-compatible codecs in order of preference for browser playback
        writer_ok = False
        for codec in ["avc1", "H264", "mp4v"]:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if out_writer.isOpened():
                append_log(job_id, f"Video codec: {codec}")
                writer_ok = True
                break
            out_writer.release()

        if not writer_ok:
            out_writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

        update_job(
            job_id,
            status="processing",
            phase="Pass 1/2 — Detection + Tracking",
            total_frames=total_frames,
            current_frame=0,
            output_file=output_filename,
            video_info={"width": width, "height": height, "fps": fps, "total_frames": total_frames},
        )
        append_log(job_id, f"Video: {width}x{height} @ {fps:.1f} fps, {total_frames} frames")
        append_log(job_id, f"Detection every {params['detect_every']} frame(s). SlowFast every {CLIP_LEN} frames.")
        append_log(job_id, "Pass 1/2 — detection + tracking + queuing SlowFast clips...")

        cap = MyVideoCapture(input_path)

        id_to_ava_labels = {}
        labels_lock = threading.Lock()
        fight_state = {"last_alert_frame": -1}
        fight_lock = threading.Lock()
        alert_threads = []
        alert_threads_lock = threading.Lock()
        fight_cooldown_frames = int(fps * params.get("fight_cooldown", FIGHT_COOLDOWN_SECS))

        clip_queue = queue.Queue()

        def slowfast_worker():
            while True:
                item = clip_queue.get()
                if item is None:
                    clip_queue.task_done()
                    break
                clip, tracked_snap, frame_idx, clip_num = item
                try:
                    person_boxes = tracked_snap[:, :4]
                    action_labels, fight_detected = run_slowfast(
                        slowfast, clip, person_boxes, ava_labels, slowfast_device, params["imsize"]
                    )
                    with labels_lock:
                        for row, label in zip(tracked_snap, action_labels):
                            id_to_ava_labels[row[5]] = label

                    if fight_detected:
                        with fight_lock:
                            frames_since = frame_idx - fight_state["last_alert_frame"]
                            if frames_since >= fight_cooldown_frames:
                                fight_state["last_alert_frame"] = frame_idx
                                do_alert = True
                            else:
                                do_alert = False

                        if do_alert:
                            msg = f"FIGHT DETECTED at frame {frame_idx}!"
                            append_log(job_id, msg)
                            with jobs_lock:
                                if job_id in jobs:
                                    jobs[job_id].setdefault("fights", []).append({
                                        "frame": frame_idx,
                                        "time": f"{frame_idx / fps:.1f}s",
                                    })
                            t = threading.Thread(
                                target=handle_fight_alert,
                                args=(cap, frame_idx),
                                daemon=False,
                            )
                            with alert_threads_lock:
                                alert_threads.append(t)
                            t.start()
                except Exception as exc:
                    append_log(job_id, f"SlowFast clip #{clip_num} failed: {exc}")
                finally:
                    clip_queue.task_done()

        worker = threading.Thread(target=slowfast_worker, daemon=False)
        worker.start()

        frame_store = []
        last_tracked = np.zeros((0, 8), dtype=np.float32)
        clip_count = 0
        t_start = time.time()

        while not cap.end:
            ret, frame = cap.read()
            if not ret:
                continue

            if cap.idx % params["detect_every"] == 0:
                boxes_xyxy, boxes_xywh, confs, classes = detect_persons(
                    yolo, frame, params["conf"], params["iou"], params["imsize"]
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
                tracked = last_tracked

            frame_store.append((frame.copy(), tracked.copy()))

            if len(cap.stack) == CLIP_LEN:
                clip_count += 1
                clip = cap.get_video_clip()
                if len(tracked) > 0:
                    clip_queue.put((clip, tracked.copy(), cap.idx, clip_count))
                    append_log(job_id, f"Clip #{clip_count} queued (frame {cap.idx}, {len(tracked)} persons)")

            if cap.idx % 30 == 0:
                progress = min(int((cap.idx / max(total_frames, 1)) * 80), 80)
                update_job(job_id, current_frame=cap.idx, progress=progress)

        clip_queue.put(None)
        worker.join()

        with alert_threads_lock:
            pending = list(alert_threads)
        for t in pending:
            t.join()

        # Pass 2 — write output video
        update_job(job_id, phase="Pass 2/2 — Writing annotated video", progress=85)
        append_log(job_id, f"Pass 2/2 — writing {len(frame_store)} annotated frames...")

        with labels_lock:
            final_labels = dict(id_to_ava_labels)

        for i, (frame, tracked) in enumerate(frame_store):
            annotated = draw_boxes(frame, tracked, final_labels)
            out_writer.write(annotated)
            if i % 100 == 0:
                progress = 85 + int((i / max(len(frame_store), 1)) * 14)
                update_job(job_id, progress=min(progress, 99))

        cap.release()
        out_writer.release()

        # Re-encode to H.264 for browser playback if needed
        import subprocess
        needs_reencode = True
        try:
            # Check if the file is already H.264 by probing with ffprobe/ffmpeg
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=codec_name", "-of", "csv=p=0",
                 output_path],
                capture_output=True, text=True, timeout=10
            )
            if "h264" in probe.stdout.lower():
                needs_reencode = False
                append_log(job_id, "Output already H.264 — no re-encoding needed.")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        if needs_reencode:
            update_job(job_id, phase="Encoding for web playback...", progress=99)
            append_log(job_id, "Re-encoding to H.264 for browser playback...")
            h264_output = output_path.replace(".mp4", "_h264.mp4")
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", output_path,
                        "-vcodec", "libx264",
                        "-preset", "fast",
                        "-crf", "23",
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                        "-an",
                        h264_output,
                    ],
                    check=True,
                    capture_output=True,
                    timeout=300,
                )
                os.replace(h264_output, output_path)
                append_log(job_id, "H.264 encoding complete.")
            except FileNotFoundError:
                append_log(job_id, "ffmpeg not found — install it for browser playback: brew install ffmpeg")
            except Exception as exc:
                append_log(job_id, f"Re-encode skipped ({exc}).")

        elapsed = time.time() - t_start
        video_secs = cap.idx / fps
        append_log(
            job_id,
            f"Done! Frames: {cap.idx} | Video: {video_secs:.1f}s | Wall time: {elapsed:.1f}s"
        )
        update_job(
            job_id,
            status="complete",
            phase="Complete",
            progress=100,
            elapsed=f"{elapsed:.1f}s",
        )

    except Exception as exc:
        log.exception(f"Job {job_id} failed")
        update_job(job_id, status="error", phase=f"Error: {exc}")
        append_log(job_id, f"ERROR: {exc}")


# ──────────────────────────────────────────────────────────
# Flask Routes
# ──────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    job_id = str(uuid.uuid4())[:8]
    ext = file.filename.rsplit(".", 1)[1].lower()
    saved_name = f"{job_id}.{ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_name)
    file.save(saved_path)

    params = {
        "yolo_model": request.form.get("yolo_model", "yolov8s.pt"),
        "reid_model": request.form.get("reid_model", "models/ckpt.t7"),
        "imsize": int(request.form.get("imsize", 640)),
        "conf": float(request.form.get("conf", 0.4)),
        "iou": float(request.form.get("iou", 0.45)),
        "detect_every": int(request.form.get("detect_every", 2)),
        "fight_cooldown": float(request.form.get("fight_cooldown", 0.0)),
        "device": request.form.get("device", "cpu"),
    }

    with jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "phase": "Queued",
            "progress": 0,
            "logs": [],
            "fights": [],
            "output_file": None,
            "input_file": file.filename,
            "total_frames": 0,
            "current_frame": 0,
            "video_info": None,
            "elapsed": None,
        }

    thread = threading.Thread(
        target=process_video,
        args=(job_id, saved_path, params),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id, "message": "Processing started"})


@app.route("/status/<job_id>")
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/output/<filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_DIR, filename, mimetype="video/mp4")


@app.route("/models/available")
def available_models():
    yolo_models = sorted(glob.glob("*.pt"))
    return jsonify({"yolo_models": yolo_models})


# ──────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Action Recognition Web UI")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    log.info(f"Starting web UI on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)

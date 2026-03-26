"""
Video capture and I/O utilities.
"""

import logging
import cv2
import torch
import numpy as np

log = logging.getLogger(__name__)


class MyVideoCapture:
    """
    A thin wrapper around cv2.VideoCapture that also maintains a rolling
    frame buffer used to feed the SlowFast model every CLIP_LEN frames.

    Attributes
    ----------
    idx   : int   — current frame index (0-based)
    end   : bool  — True once the source is exhausted
    stack : list  — rolling buffer of raw BGR frames (numpy arrays)
    """

    def __init__(self, source):
        """
        Parameters
        ----------
        source : str | int
            Video file path or camera index (e.g. 0 for webcam).
        """
        self.filename = source
        self.cap      = cv2.VideoCapture(source)
        self.idx      = -1
        self.end      = False
        self.stack    = []

        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")

    # ── Frame reading ──────────────────────────────────────

    def read(self):
        """
        Read the next frame and append it to the internal buffer.

        Returns
        -------
        ret   : bool     — False when the source is exhausted
        frame : ndarray  — BGR frame (or None on failure)
        """
        self.idx += 1
        ret, frame = self.cap.read()
        if ret:
            self.stack.append(frame)
        else:
            self.end = True
        return ret, frame

    # ── Tensor conversion ──────────────────────────────────

    @staticmethod
    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        """Convert a single BGR frame (H, W, 3) to an RGB uint8 tensor (1, H, W, 3)."""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).unsqueeze(0)

    def get_video_clip(self) -> torch.Tensor:
        """
        Convert the current frame buffer to a (C, T, H, W) float tensor
        and reset the buffer for the next clip.

        Returns
        -------
        Tensor (C, T, H, W) — where C=3, T=len(stack)
        """
        assert len(self.stack) > 0, "Frame buffer is empty — nothing to convert."
        tensors   = [self._to_tensor(f) for f in self.stack]
        clip      = torch.cat(tensors)          # (T, H, W, 3)
        clip      = clip.permute(3, 0, 1, 2)   # (C, T, H, W)
        self.stack = []                         # reset buffer
        return clip

    # ── Clip extraction for alerts ─────────────────────────

    def get_frames_around_index(self, index: int, frame_buffer: int) -> list:
        """
        Re-open the source video and extract frames centred on *index*.

        Parameters
        ----------
        index        : int — central frame number
        frame_buffer : int — half-window size (total = 2 * frame_buffer + 1)

        Returns
        -------
        list[ndarray] — BGR frames within bounds
        """
        if isinstance(self.filename, int):
            log.warning("Cannot extract alert clip from a live camera.")
            return []

        cap = cv2.VideoCapture(self.filename)
        if not cap.isOpened():
            log.error(f"Cannot re-open video for alert clip: {self.filename}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = max(0, index - frame_buffer)
        end   = min(total_frames - 1, index + frame_buffer)

        frames = []
        for i in range(start, end + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                log.warning(f"Could not read frame {i} for alert clip.")
        cap.release()
        return frames

    # ── Cleanup ────────────────────────────────────────────

    def release(self):
        """Release the underlying cv2.VideoCapture."""
        self.cap.release()


# ──────────────────────────────────────────────────────────────
# Video saving
# ──────────────────────────────────────────────────────────────

def save_video(frame_list: list, dst: str, fps: float = 25.0) -> bool:
    """
    Write a list of BGR frames to an MP4 video file.

    Parameters
    ----------
    frame_list : list[ndarray] — BGR frames to write
    dst        : str           — output file path (must end in .mp4)
    fps        : float         — frames per second for the output video

    Returns
    -------
    bool — True on success, False on error
    """
    if not frame_list:
        log.error("save_video: empty frame list — nothing to write.")
        return False

    h, w = frame_list[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(dst, fourcc, fps, (w, h))

    if not writer.isOpened():
        log.error(f"save_video: cannot open output file: {dst}")
        return False

    for frame in frame_list:
        writer.write(frame)
    writer.release()
    log.info(f"Clip saved → {dst}  ({len(frame_list)} frames @ {fps} fps)")
    return True

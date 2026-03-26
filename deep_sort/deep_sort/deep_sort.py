import numpy as np
import torch

from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker


__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=3.0, max_iou_distance=0.7, max_age=70, n_init=2, nn_budget=100, use_cuda=True, use_appearence=True):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        self.use_appearence=use_appearence
        self.extractor = Extractor(model_path, use_cuda=use_cuda)

        max_cosine_distance = max_dist
        nn_budget = nn_budget
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, labels, ori_img):
        """
        Update tracker with new detections for the current frame.

        Parameters
        ----------
        bbox_xywh   : Tensor/ndarray (N, 4) — center_x, center_y, w, h
        confidences : Tensor/ndarray (N, 1) — detection confidence scores
        labels      : list[int]             — class IDs per detection
        ori_img     : ndarray (H, W, 3)     — RGB image (used for ReID crop)

        Returns
        -------
        outputs : ndarray (M, 8) — [x1, y1, x2, y2, class, track_id, vx, vy]
                  One row per confirmed active track.
        """
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        
        if self.use_appearence:
            features = self._get_features(bbox_xywh, ori_img)
        else:
            features = np.array([np.array([0.5,0.5]) for _ in range(len(bbox_xywh))])
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        # Use float(conf) to safely convert Tensor scalars (shape [1]) to Python
        # floats before comparing — avoids ambiguous Tensor boolean evaluation.
        detections = [
            Detection(bbox_tlwh[i], float(conf), labels[i], features[i])
            for i, conf in enumerate(confidences)
            if float(conf) > self.min_confidence
        ]

        # run on non-maximum supression
        # boxes = np.array([d.tlwh for d in detections])
        # scores = np.array([d.confidence for d in detections])
        # indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        # detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            label=track.label
            Vx=10*track.mean[4]
            Vy=10*track.mean[5]
            outputs.append(np.array([x1,y1,x2,y2,label,track_id,Vx,Vy], dtype=np.int32))
        if len(outputs) > 0:
            outputs = np.stack(outputs,axis=0)
        return outputs


    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        """
        Convert bounding boxes from center format (cx, cy, w, h)
        to top-left format (x_tl, y_tl, w, h).
        """
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.0
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.0
        return bbox_tlwh


    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        Convert bounding box from top-left format (x_tl, y_tl, w, h)
        to corner format (x1, y1, x2, y2), clipped to image boundaries.
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        """Convert corner format (x1,y1,x2,y2) to top-left format (x_tl,y_tl,w,h)."""
        x1, y1, x2, y2 = bbox_xyxy
        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h
    
    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2,x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features



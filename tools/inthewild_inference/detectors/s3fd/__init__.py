import os
import time
import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .nets import S3FDNet
from .box_utils import decode, nms_

logger = logging.getLogger(__name__)

PATH_WEIGHT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'weights', 'checkpoints', 's3fd-619a316812.pth',
)
IMG_MEAN = np.array([104.0, 117.0, 123.0], dtype=np.float32)


class S3FD:

    def __init__(self, device='cuda'):
        tstamp = time.time()
        self.device = device

        logger.info('[S3FD] loading with %s', self.device)
        self.net = S3FDNet().to(self.device)
        state_dict = torch.load(PATH_WEIGHT, map_location=self.device, weights_only=True)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        logger.info('[S3FD] finished loading (%.4f sec)', time.time() - tstamp)

    def _detect_one(self, image_bgr, conf_th):
        """Run S3FD on a single BGR image. Returns (N, 5) boxes in image coords."""
        img = image_bgr.astype(np.float32) - IMG_MEAN
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            olist = self.net(img)

        bboxlist = []
        for i in range(len(olist) // 2):
            olist[i * 2] = F.softmax(olist[i * 2], dim=1)

        olist = [oelem.data.cpu() for oelem in olist]
        for i in range(len(olist) // 2):
            ocls, oreg = olist[i * 2], olist[i * 2 + 1]
            stride = 2 ** (i + 2)  # 4, 8, 16, 32, 64, 128
            poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
            for _, hindex, windex in poss:
                axc = stride / 2 + windex * stride
                ayc = stride / 2 + hindex * stride
                score = float(ocls[0, 1, hindex, windex])
                loc = oreg[0, :, hindex, windex].contiguous().view(1, 4)
                priors = torch.Tensor([[axc, ayc, stride * 4, stride * 4]])
                box = decode(loc, priors, [0.1, 0.2])
                x1, y1, x2, y2 = box[0].tolist()
                bboxlist.append([x1, y1, x2, y2, score])

        if not bboxlist:
            return np.empty((0, 5), dtype=np.float32)

        bboxes = np.array(bboxlist, dtype=np.float32)
        keep = nms_(bboxes, 0.3)
        bboxes = bboxes[keep]
        bboxes = bboxes[bboxes[:, -1] > conf_th]
        return bboxes

    def detect_faces(self, image, conf_th=0.8, scales=[1]):
        """Detect faces in an RGB image.

        Returns ndarray of shape (N, 5): [x1, y1, x2, y2, score] in original
        image coordinates.
        """
        # Checkpoint / Wav2Lip pipeline expects BGR + ImageNet-style mean.
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        bboxes = np.empty((0, 5), dtype=np.float32)

        for s in scales:
            if s == 1:
                scaled = image_bgr
            else:
                scaled = cv2.resize(
                    image_bgr, dsize=(0, 0), fx=s, fy=s,
                    interpolation=cv2.INTER_LINEAR,
                )

            dets = self._detect_one(scaled, conf_th=conf_th)
            if len(dets) == 0:
                continue

            if s != 1:
                dets = dets.copy()
                dets[:, :4] /= s

            bboxes = np.vstack((bboxes, dets)) if len(bboxes) else dets

        if len(bboxes) == 0:
            return bboxes

        keep = nms_(bboxes, 0.1)
        return bboxes[keep]

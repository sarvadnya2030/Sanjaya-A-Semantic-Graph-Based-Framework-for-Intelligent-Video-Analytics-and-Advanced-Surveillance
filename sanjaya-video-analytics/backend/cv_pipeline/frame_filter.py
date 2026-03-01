import cv2
import numpy as np

class FrameFilter:
    def __init__(self, ssim_thresh=0.98, hist_thresh=0.92, min_step=3):
        self.ssim_thresh = ssim_thresh
        self.hist_thresh = hist_thresh
        self.prev_kept_gray = None
        self.prev_kept_hsv_hist = None
        self.frame_since_keep = 0
        self.min_step = min_step  # always allow at least every Nth frame

    def _luma(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _hist_hsv(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv],[0,1],None,[32,32],[0,180,0,256])
        cv2.normalize(hist, hist)
        return hist

    def _ssim(self, a, b):
        diff = cv2.absdiff(a, b)
        score = 1.0 - (float(np.mean(diff)) / 255.0)
        return max(0.0, min(1.0, score))

    def keep(self, frame):
        gray = self._luma(frame)
        hist = self._hist_hsv(frame)

        if self.prev_kept_gray is None or self.prev_kept_hsv_hist is None:
            self.prev_kept_gray = gray
            self.prev_kept_hsv_hist = hist
            self.frame_since_keep = 0
            return True

        self.frame_since_keep += 1
        if self.frame_since_keep >= self.min_step:
            ssim = self._ssim(gray, self.prev_kept_gray)
            hsim = cv2.compareHist(self.prev_kept_hsv_hist, hist, cv2.HISTCMP_CORREL)
            if ssim >= self.ssim_thresh and hsim >= self.hist_thresh:
                return False
            self.prev_kept_gray = gray
            self.prev_kept_hsv_hist = hist
            self.frame_since_keep = 0
            return True

        hsim = cv2.compareHist(self.prev_kept_hsv_hist, hist, cv2.HISTCMP_CORREL)
        if hsim < 0.75:
            self.prev_kept_gray = gray
            self.prev_kept_hsv_hist = hist
            self.frame_since_keep = 0
            return True

        return False
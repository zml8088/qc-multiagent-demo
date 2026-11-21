# -*- coding: utf-8 -*-
"""
感知层：经典 CV 异常检测（充当"检测仪器"，无需 GPU，替代 PatchCore 的角色）。

流程：
  1) 与标准件模板差分 -> 异常热力图 -> 可疑像素掩膜
  2) 连通域分析 -> 可疑区域（面积/长宽比/峰值强度/位置）
  3) 轮廓几何比对（先闭运算填补细划痕，再取外轮廓）-> 形状畸变特征
输出结构化感知报告，供上层 Agent 判定全部依据。
"""
import cv2
import numpy as np

from .synthetic_data import IMG_SIZE, generate_one

DIFF_THRESHOLD = 18     # 差分阈值
MIN_REGION_AREA = 10    # 最小可疑区域面积


def _make_template(part, n=9, seed=123):
    """多张良品图取中值构造标准件模板（模拟产线标准样件建模）；不加光照抖动保证模板纯净"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    stack = []
    for _ in range(n):
        img, _ = generate_one(part, defect="none", jitter=False)
        stack.append(img.astype(np.float32))
    return np.median(np.stack(stack), axis=0).astype(np.uint8)


def _contour_stats(im):
    """工件主体轮廓的几何特征。先闭运算桥接细划痕造成的断裂，避免面积被误判。"""
    binary = (im > 110).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    hull_area = cv2.contourArea(cv2.convexHull(c)) or 1.0
    return {"area": float(area), "bbox": (w, h), "ar": w / max(h, 1), "solidity": area / hull_area}


class PerceptionModule:
    def __init__(self):
        self.templates = {p: _make_template(p) for p in ("screw", "nut")}
        self._tmpl_stats = {p: _contour_stats(t) for p, t in self.templates.items()}

    def analyze(self, img, part):
        tmpl = self.templates[part]

        # ---- 差分 ----
        diff = cv2.absdiff(img, tmpl)
        blur = cv2.GaussianBlur(diff, (5, 5), 0)
        _, mask = cv2.threshold(blur, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        # ---- 连通域 ----
        num, lbl, stats, cents = cv2.connectedComponentsWithStats(mask)
        regions = []
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < MIN_REGION_AREA:
                continue
            x, y, w, h = (int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                     cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
            regions.append({
                "area": area,
                "bbox": [x, y, w, h],
                "elongation": round(max(w, h) / max(1, min(w, h)), 2),
                "peak_intensity": int(blur[lbl == i].max()),
            })
        regions.sort(key=lambda r: -r["area"])
        anomaly_score = round(float((mask > 0).sum()) / mask.size * 100, 3)

        # ---- 几何畸变 ----
        geo = self._geo_distort(img, self._tmpl_stats[part])

        return {
            "part": part,
            "anomaly_score_pct": anomaly_score,
            "n_suspicious_regions": len(regions),
            "max_region_area": regions[0]["area"] if regions else 0,
            "max_elongation": max([r["elongation"] for r in regions], default=0.0),
            "regions": regions[:4],
            "geo_distortion": geo["score"],
            "area_diff_signed_pct": geo["area_diff_signed_pct"],
            "ar_diff": geo["ar_diff"],
            "solidity_delta": geo["solidity_delta"],
        }

    def _geo_distort(self, img, tstat):
        s = _contour_stats(img)
        if s is None or tstat is None:
            return {"score": 99.0, "area_diff_signed_pct": 99.0, "ar_diff": 9.9, "solidity_delta": 0.9}
        area_diff_signed = (s["area"] - tstat["area"]) / max(tstat["area"], 1) * 100
        ar_diff = abs(s["ar"] - tstat["ar"])
        sol_delta = abs(s["solidity"] - tstat["solidity"])
        score = abs(area_diff_signed) + ar_diff * 10 + sol_delta * 120
        return {
            "score": round(float(score), 2),
            "area_diff_signed_pct": round(float(area_diff_signed), 2),
            "ar_diff": round(float(ar_diff), 3),
            "solidity_delta": round(float(sol_delta), 3),
        }

    def heatmap(self, img, part):
        tmpl = self.templates[part]
        blur = cv2.GaussianBlur(cv2.absdiff(img, tmpl), (5, 5), 0)
        heat = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    def overlay(self, img, part):
        base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for r in self.analyze(img, part)["regions"]:
            x, y, w, h = r["bbox"]
            cv2.rectangle(base, (x, y), (x + w, y + h), (0, 0, 255), 2)
        return base

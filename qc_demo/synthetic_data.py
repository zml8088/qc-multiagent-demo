# -*- coding: utf-8 -*-
"""
合成螺丝/螺母缺陷图像生成器（夹具定位版：工件朝向固定，便于与模板像素级对齐）。

图像 256x256 灰度工业风：
  - screw：头部矩形 + 一字槽 + 带螺纹的杆身（x∈[109,147], y∈[62,222]）
  - nut  ：六角形本体 + 中心圆孔（中心 128,128，外接圆半径 62）

缺陷：scratch(划痕) / bent(变形) / missing(缺料) / rust(锈蚀)
严重度：mild / moderate / critical
另叠加全图光照抖动，模拟产线光源波动（这是良品也会产生轻微异常分的真实原因）。
"""
import cv2
import numpy as np
import random

IMG_SIZE = 256
DEFECTS = ["scratch", "bent", "missing", "rust"]          # 螺丝适用
DEFECTS_BY_PART = {
    "screw": ["scratch", "bent", "missing", "rust"],
    "nut": ["scratch", "color", "missing", "rust"],        # 螺母用色差替代弯曲（2D 下弯曲与缺料不可分）
}
SEVERITIES = ["mild", "moderate", "critical"]
SEV2GRADE = {"mild": "OK", "moderate": "REWORK", "critical": "SCRAP"}
SEV_IDX = {"mild": 1, "moderate": 2, "critical": 3}

BG_VAL = 58
PART_VAL = 190

# 螺丝杆身几何
SHAFT_TOP, SHAFT_BOT, SHAFT_W = 62, 222, 38
CX = IMG_SIZE // 2


def _bg():
    img = np.full((IMG_SIZE, IMG_SIZE), BG_VAL, dtype=np.float32)
    img += np.linspace(-6, 6, IMG_SIZE, dtype=np.float32)[None, :]
    img += np.random.normal(0, 1.8, (IMG_SIZE, IMG_SIZE))
    return np.clip(img, 0, 255).astype(np.uint8)


def _lighting_jitter(img):
    """产线光源波动：整体亮度/对比度轻微变化"""
    gain = 1.0 + np.random.normal(0, 0.018)
    bias = np.random.normal(0, 5.0)
    return np.clip(img.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)


def _draw_screw(img, bend=None):
    head_w, head_h, head_y = 84, 26, 36
    pts = np.array([[CX - head_w // 2, head_y], [CX + head_w // 2, head_y],
                    [CX + head_w // 2 - 10, head_y + head_h], [CX - head_w // 2 + 10, head_y + head_h]])
    cv2.fillPoly(img, [pts], PART_VAL)
    cv2.line(img, (CX - 22, head_y + 10), (CX + 22, head_y + 10), 120, 4)

    if bend is None:
        cv2.rectangle(img, (CX - SHAFT_W // 2, SHAFT_TOP), (CX + SHAFT_W // 2, SHAFT_BOT), PART_VAL, -1)
        for y in range(SHAFT_TOP + 5, SHAFT_BOT - 3, 9):
            cv2.line(img, (CX - SHAFT_W // 2 + 3, y), (CX + SHAFT_W // 2 - 3, y), 140, 2)
    else:
        ys = [SHAFT_TOP, (SHAFT_TOP + SHAFT_BOT) // 2, SHAFT_BOT]
        left = [(CX - SHAFT_W // 2, ys[0]), (CX - SHAFT_W // 2 + bend, ys[1]), (CX - SHAFT_W // 2, ys[2])]
        right = [(CX + SHAFT_W // 2, ys[0]), (CX + SHAFT_W // 2 + bend, ys[1]), (CX + SHAFT_W // 2, ys[2])]
        poly = np.array(left + right[::-1], dtype=np.int32)
        cv2.fillPoly(img, [poly], PART_VAL)
        for i in range(2):
            for t in (0.4, 0.8):
                x = int(left[i][0] + (left[i + 1][0] - left[i][0]) * t)
                x2 = int(right[i][0] + (right[i + 1][0] - right[i][0]) * t)
                y = int(ys[i] + (ys[i + 1] - ys[i]) * t)
                cv2.line(img, (x + 3, y), (x2 - 3, y), 140, 2)


def _draw_nut(img, deform=0):
    ang0 = np.pi / 6
    R = 62
    verts = []
    for i in range(6):
        a = ang0 + i * np.pi / 3
        r = R - deform if (deform and i == 1) else R
        verts.append([int(CX + r * np.cos(a)), int(CX + r * np.sin(a))])
    pts = np.array(verts, dtype=np.int32)
    cv2.fillPoly(img, [pts], PART_VAL)
    cv2.polylines(img, [pts], True, 145, 2)
    cv2.circle(img, (CX, CX), 26, BG_VAL, -1)
    cv2.circle(img, (CX, CX), 26, 120, 2)


def _add_scratch(img, part, sev):
    """划痕画在工件表面。螺丝用纵向划痕（贴合杆身），螺母用斜向划痕。"""
    i = SEV_IDX[sev]
    n = {1: 1, 2: 2, 3: 3}[i]
    length = {1: 30, 2: 52, 3: 76}[i]
    thick = {1: 2, 2: 3, 3: 4}[i]
    for _ in range(n):
        if part == "screw":
            x = CX + random.randint(-SHAFT_W // 2 + 4, SHAFT_W // 2 - 4)
            y1 = random.randint(SHAFT_TOP + 12, SHAFT_BOT - length - 6)
            cv2.line(img, (x, y1), (x + random.randint(-4, 4), y1 + length), 38, thick)
        else:
            y = random.randint(CX - 50, CX + 50)
            x1 = random.randint(CX - 46, CX + 46 - length)
            if abs(y - CX) < 28:      # 避开中心孔
                y = CX + random.choice([-1, 1]) * random.randint(30, 50)
            cv2.line(img, (x1, y), (x1 + length, y + random.randint(-4, 4)), 38, thick)


def _add_rust(img, part, sev):
    i = SEV_IDX[sev]
    n = {1: 11, 2: 26, 3: 46}[i]
    rmax = {1: 4, 2: 6, 3: 8}[i]
    for _ in range(n):
        if part == "screw":
            x = CX + random.randint(-14, 14)
            y = random.randint(SHAFT_TOP + 10, SHAFT_BOT - 10)
        else:
            a = random.uniform(0, 2 * np.pi)
            rr = random.uniform(30, 52)
            x, y = int(CX + rr * np.cos(a)), int(CX + rr * np.sin(a))
        cv2.circle(img, (x, y), random.randint(2, rmax), 125, -1)


def _add_missing(img, part, sev):
    """缺料挖在工件边缘（螺丝挖杆身侧边，螺母挖六角边缘）"""
    i = SEV_IDX[sev]
    r = {1: 9, 2: 16, 3: 25}[i]
    if part == "screw":
        side = random.choice([-1, 1])
        px = CX + side * SHAFT_W // 2
        py = random.randint(SHAFT_TOP + 40, SHAFT_BOT - 40)
    else:
        a = random.uniform(0, 2 * np.pi)
        px, py = int(CX + 56 * np.cos(a)), int(CX + 56 * np.sin(a))
    cv2.circle(img, (px, py), r, BG_VAL, -1)


def _add_color(img, part, sev):
    """色差：工件表面的局部发暗/发亮区域（软边缘大色块）"""
    i = SEV_IDX[sev]
    r = {1: 13, 2: 20, 3: 28}[i]
    for _ in range({1: 1, 2: 2, 3: 2}[i]):
        if part == "screw":
            x = CX + random.randint(-10, 10)
            y = random.randint(SHAFT_TOP + 30, SHAFT_BOT - 30)
        else:
            a = random.uniform(0, 2 * np.pi)
            rr = random.uniform(30, 40)            # 落在六角环形本体上，不侵犯中心孔
            x, y = int(CX + rr * np.cos(a)), int(CX + rr * np.sin(a))
        overlay = img.copy()
        cv2.circle(overlay, (x, y), r, 158, -1)     # 表面发暗，模拟色差
        overlay = cv2.GaussianBlur(overlay, (15, 15), 0)
        cv2.addWeighted(overlay, 0.80, img, 0.20, 0, img)


def _add_texture(img, part):
    """机加工纹理：表面浅凹槽（斜向短小），良品也会有，构成与细划痕的真实歧义"""
    for _ in range(random.randint(2, 3)):
        ang = random.uniform(np.pi / 9, np.pi / 2.6)      # 20°~69°，斜向
        L = random.randint(9, 15)
        if part == "screw":
            x = CX + random.randint(-9, 9)
            y = random.randint(SHAFT_TOP + 32, SHAFT_BOT - 32)
        else:
            a = random.uniform(0, 2 * np.pi)
            rr = random.uniform(0, 28)
            x, y = int(CX + rr * np.cos(a)), int(CX + rr * np.sin(a))
        cv2.line(img, (x, y), (int(x + L * np.cos(ang)), int(y + L * np.sin(ang))), 158, 2)


BEND_PARAM = {"mild": 3, "moderate": 9, "critical": 16}
DEFORM_PARAM = {"mild": 3, "moderate": 7, "critical": 12}


def generate_one(part="screw", defect=None, severity=None, good_ratio=0.3, jitter=True):
    img = _bg()
    label = {"part": part, "defect": "none", "severity": "none", "grade": "OK"}

    allowed = DEFECTS_BY_PART.get(part, DEFECTS)
    if defect is not None and defect not in allowed and defect != "none":
        defect = random.choice(allowed)
    if defect is None:
        defect = "none" if random.random() < good_ratio else random.choice(allowed)
    if defect == "none":
        severity = "none"
    else:
        severity = severity or random.choice(SEVERITIES)
        label.update({"defect": defect, "severity": severity, "grade": SEV2GRADE[severity]})

    if part == "screw":
        _draw_screw(img, bend=BEND_PARAM[severity] if defect == "bent" else None)
        if defect == "scratch":
            _add_scratch(img, part, severity)
        elif defect == "rust":
            _add_rust(img, part, severity)
        elif defect == "missing":
            _add_missing(img, part, severity)
        elif defect == "color":
            _add_color(img, part, severity)
    else:
        _draw_nut(img, deform=DEFORM_PARAM[severity] if defect == "bent" else 0)
        if defect == "scratch":
            _add_scratch(img, part, severity)
        elif defect == "rust":
            _add_rust(img, part, severity)
        elif defect == "missing":
            _add_missing(img, part, severity)
        elif defect == "color":
            _add_color(img, part, severity)

    _add_texture(img, part)
    if jitter:
        img = _lighting_jitter(img)
    return img, label


def generate_dataset(n=120, seed=0, good_ratio=0.3):
    random.seed(seed)
    np.random.seed(seed)
    imgs, labels = [], []
    for i in range(n):
        part = "screw" if i % 2 == 0 else "nut"
        img, lab = generate_one(part, good_ratio=good_ratio)
        imgs.append(img)
        labels.append(lab)
    return imgs, labels


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    imgs, labels = generate_dataset(6, seed=42, good_ratio=0.5)
    for i, (im, lb) in enumerate(zip(imgs, labels)):
        cv2.imwrite(f"data/sample_{i}_{lb['part']}_{lb['defect']}_{lb['severity']}.png", im)
        print(lb)

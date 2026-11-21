# -*- coding: utf-8 -*-
"""把合成样本导出成图片文件，方便查看/放进 PPT。

用法：
    python tools/make_samples.py                # 默认导出到 samples/
    python tools/make_samples.py <outdir> <seed>

产出：
    <outdir>/*.png                 单张样本（256×256 灰度）
    <outdir>/grid.png              总览拼图（2 行 × 13 列）
    <outdir>/labels.csv            每张图的真值标注
"""
import csv
import os
import sys

# 允许从任意目录运行：把项目根目录加入模块搜索路径
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qc_demo.synthetic_data import generate_one, DEFECTS_BY_PART, SEVERITIES, SEV2GRADE

OUT = sys.argv[1] if len(sys.argv) > 1 else "samples"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 2026


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, grid = [], {}

    for part in ("screw", "nut"):
        cats = ["none"] + list(DEFECTS_BY_PART[part])
        grid[part] = {}
        for defect in cats:
            sevs = ["none"] if defect == "none" else list(SEVERITIES)
            for sev in sevs:
                np.random.seed(SEED)
                img, lab = generate_one(part, defect, sev)
                name = f"{part}__{defect}__{sev}.png"
                cv2.imwrite(os.path.join(OUT, name), img)
                rows.append([name, part, defect, sev, lab.get("grade", "OK")])
                grid[part].setdefault(defect, {})[sev] = img
        # 良品只有一档，占位补齐
        grid[part]["none"] = {"mild": grid[part]["none"]["none"],
                              "moderate": grid[part]["none"]["none"],
                              "critical": grid[part]["none"]["none"]}

    with open(os.path.join(OUT, "labels.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "part", "defect", "severity", "grade"])
        w.writerows(rows)

    # 总览拼图：行=件型，列= 良品 + 4 类缺陷 × 3 个严重度
    parts = ["screw", "nut"]
    cols = ["none"] + [d for d in DEFECTS_BY_PART["screw"]]
    fig, axes = plt.subplots(len(parts), 13, figsize=(26, 4.6))
    for ri, part in enumerate(parts):
        ci = 0
        axes[ri, ci].imshow(grid[part]["none"]["mild"], cmap="gray")
        axes[ri, ci].set_title(f"{part}\ngood", fontsize=9)
        ci += 1
        for defect in DEFECTS_BY_PART[part]:
            for sev in SEVERITIES:
                ax = axes[ri, ci]
                ax.imshow(grid[part][defect][sev], cmap="gray")
                ax.set_title(f"{part}\n{defect}/{sev}", fontsize=9)
                ci += 1
        for ax in axes[ri]:
            ax.axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "grid.png"), dpi=120)
    print("导出目录:", os.path.abspath(OUT))
    print("图片数量:", len(rows), "| 总览:", os.path.join(OUT, "grid.png"))
    print("真值表:", os.path.join(OUT, "labels.csv"))


if __name__ == "__main__":
    main()

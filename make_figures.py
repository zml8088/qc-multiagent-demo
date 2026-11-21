# -*- coding: utf-8 -*-
"""
生成报告用图：figures/fig1_optimization.png（优化曲线）
              figures/fig2_samples.png（样本与热力图）
              figures/fig3_feature_space.png（特征空间可分性）
用法： python make_figures.py
"""
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qc_demo.synthetic_data import generate_dataset, generate_one, DEFECTS_BY_PART
from qc_demo.perception import PerceptionModule
from qc_demo.pipeline import prepare, optimize_loop, evaluate_points, QCPipeline, baseline_single_threshold

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
os.makedirs("figures", exist_ok=True)


def fig_optimization():
    dev_imgs, dev_labels = generate_dataset(140, seed=7)
    test_imgs, test_labels = generate_dataset(160, seed=99)
    dev = prepare(dev_imgs, dev_labels)
    test = prepare(test_imgs, test_labels)

    pipe, hist = optimize_loop(dev_imgs, dev_labels, rounds=10, verbose=False, points=dev)
    xs = [h["round"] for h in hist]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(xs, [h["metrics"]["漏检率 FNR"] * 100 for h in hist], "o-", color="#c0392b", label="漏检率")
    ax[0].plot(xs, [h["metrics"]["过杀率 FPR"] * 100 for h in hist], "s-", color="#e67e22", label="过杀率")
    ax[0].set_xlabel("优化轮次")
    ax[0].set_ylabel("%")
    ax[0].set_title("检测指标随提示词优化的变化（dev 集）")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(xs, [h["metrics"]["类型准确率"] * 100 for h in hist], "^-", color="#2c7fb8", label="缺陷类型准确率")
    ax[1].plot(xs, [h["metrics"]["分级一致率"] * 100 for h in hist], "d-", color="#27ae60", label="分级一致率")
    ax[1].set_xlabel("优化轮次")
    ax[1].set_ylabel("%")
    ax[1].set_ylim(40, 102)
    ax[1].set_title("决策指标随提示词优化的变化（dev 集）")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/fig1_optimization.png", dpi=160)
    plt.close(fig)

    m0, _ = evaluate_points(QCPipeline(), test)
    m1, _ = evaluate_points(pipe, test)
    b = baseline_single_threshold(dev, test)
    print("test 优化前:", m0)
    print("test 优化后:", m1)
    print("基线:", b)
    return pipe


def fig_samples():
    perc = PerceptionModule()
    parts = ["screw", "nut"]
    defects = {p: ["none"] + DEFECTS_BY_PART[p] for p in parts}
    rows = sum(len(v) for v in defects.values())
    fig, axes = plt.subplots(rows, 4, figsize=(7.2, 1.55 * rows))
    r = 0
    for part in parts:
        for d in defects[part]:
            sev = "none" if d == "none" else "critical"
            img, lab = generate_one(part, d, sev)
            rep = perc.analyze(img, part)
            heat = cv2.cvtColor(perc.heatmap(img, part), cv2.COLOR_BGR2RGB)
            ov = cv2.cvtColor(perc.overlay(img, part), cv2.COLOR_BGR2RGB)
            for c, (im, ttl) in enumerate([(img, f"{part}/{d}"),
                                           (heat, "异常热力图"),
                                           (ov, "可疑区域"),
                                           (None, None)]):
                ax = axes[r][c]
                ax.axis("off")
                if im is not None:
                    ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
                    ax.set_title(ttl, fontsize=8)
                else:
                    ax.text(0.02, 0.5, f"score={rep['anomaly_score_pct']:.2f}%\n"
                                       f"geo={rep['geo_distortion']:.1f}\n"
                                       f"nreg={rep['n_suspicious_regions']}\n"
                                       f"elong={rep['max_elongation']:.1f}",
                            fontsize=7.5, va="center", family="monospace")
            r += 1
    fig.suptitle("合成样本 / 感知层输出（每行：原图、热力图、可疑区域、特征值）", fontsize=10)
    fig.tight_layout()
    fig.savefig("figures/fig2_samples.png", dpi=170)
    plt.close(fig)


def fig_feature_space():
    perc = PerceptionModule()
    pts = []
    for part in ("screw", "nut"):
        for d in ["none"] + DEFECTS_BY_PART[part]:
            for _ in range(14):
                img, lab = generate_one(part, d, None)
                rep = perc.analyze(img, part)
                pts.append((rep["anomaly_score_pct"], rep["max_elongation"],
                            rep["n_suspicious_regions"], d))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"none": "#95a5a6", "scratch": "#c0392b", "rust": "#8e44ad",
              "missing": "#2980b9", "bent": "#e67e22", "color": "#27ae60"}
    for name in colors:
        sel = [(s, e) for s, e, _, d in pts if d == name]
        if sel:
            ax[0].scatter([x for x, _ in sel], [y for _, y in sel], s=22, alpha=0.8,
                          label=name, color=colors[name])
    ax[0].set_xlabel("异常分数 anomaly_score (%)")
    ax[0].set_ylabel("最大区域长宽比 max_elongation")
    ax[0].set_title("异常分数 × 长宽比：良品与细划痕重叠，靠形态可分")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    for name in colors:
        sel = [(n, s) for s, _, n, d in pts if d == name]
        if sel:
            ax[1].scatter([x for x, _ in sel], [y for _, y in sel], s=22, alpha=0.8,
                          label=name, color=colors[name])
    ax[1].set_xlabel("可疑区域数 n_suspicious_regions")
    ax[1].set_ylabel("异常分数 anomaly_score (%)")
    ax[1].set_title("区域数 × 异常分数：锈蚀（多斑）与其他类型可分")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/fig3_feature_space.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    fig_optimization()
    fig_samples()
    fig_feature_space()
    print("图表已生成到 figures/")

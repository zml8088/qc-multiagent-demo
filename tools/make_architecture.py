# -*- coding: utf-8 -*-
"""生成系统架构图 figures/fig0_architecture.png。

用法：
    python tools/make_architecture.py
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "figures", "fig0_architecture.png")

# 主题色（浅色底、深色字）
C_INPUT = "white"
C_PERCEPT = "white"
C_AGENT = "white"
C_REFLECT = "white"
C_EDGE = "#55606e"


def box(ax, x, y, w, h, title, lines, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.012",
                                linewidth=1.2, edgecolor=C_EDGE, facecolor=fc))
    ax.text(x + w / 2, y + h - 0.045, title, ha="center", va="top",
            fontsize=11.5, fontweight="bold", color="#1f2733")
    ax.text(x + w / 2, y + (h - 0.075) / 2, "\n".join(lines),
            ha="center", va="center", fontsize=9.2, color="#3a4450", linespacing=1.5)


def arrow(ax, p1, p2, style="-|>", color="#4a5561", lw=1.6, conn="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14,
                                 linewidth=lw, color=color, connectionstyle=conn))


def main():
    fig, ax = plt.subplots(figsize=(11.2, 4.9), dpi=160)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.0, 0.74)
    ax.axis("off")

    # 主链：输入 -> 感知层 -> 缺陷分析 Agent -> 分级决策 Agent -> 处置输出
    box(ax, 0.015, 0.40, 0.145, 0.22, "输入图像",
        ["合成数据（含真值）", "或产线实拍图"], C_INPUT)
    box(ax, 0.185, 0.40, 0.20, 0.22, "感知层（经典 CV）",
        ["模板差分 + 连通域", "轮廓几何特征", "anomaly_score"], C_PERCEPT)
    box(ax, 0.415, 0.40, 0.20, 0.22, "缺陷分析 Agent",
        ["提示词 RULES 判据树", "判定缺陷类型与严重度"], C_AGENT)
    box(ax, 0.645, 0.40, 0.20, 0.22, "分级决策 Agent",
        ["映射产线 SOP 处置", "报废 / 返修 / 放行", "严重度硬约束兜底"], C_AGENT)
    box(ax, 0.875, 0.40, 0.115, 0.22, "输出",
        ["质检结论", "处置建议"], C_INPUT)

    arrow(ax, (0.160, 0.51), (0.185, 0.51))
    arrow(ax, (0.385, 0.51), (0.415, 0.51))
    arrow(ax, (0.615, 0.51), (0.645, 0.51))
    arrow(ax, (0.845, 0.51), (0.875, 0.51))

    # 反思优化 Agent（在主链下方，回灌箭头指回缺陷分析 Agent）
    box(ax, 0.415, 0.05, 0.43, 0.24, "反思优化 Agent",
        ["① 误判投票归因：定位出错的判据", "② 用 dev 集标注反推判据边界",
         "③ 改写提示词 RULES 段，keep-best 回灌"], C_REFLECT)

    # 分级决策 -> 反思：误判样本
    arrow(ax, (0.745, 0.40), (0.745, 0.29), conn="arc3,rad=0")
    ax.text(0.758, 0.345, "误判样本\n（与 dev 标注比对）", fontsize=8.6,
            color="#8a5a16", ha="left", va="center", linespacing=1.4)
    # 反思 -> 缺陷分析：RULES 回灌
    arrow(ax, (0.515, 0.29), (0.515, 0.40), conn="arc3,rad=0")
    ax.text(0.528, 0.345, "优化后提示词", fontsize=8.6, color="#8a5a16",
            ha="left", va="center")
    # dev 标注数据进入反思
    box(ax, 0.185, 0.05, 0.20, 0.24, "dev 集标注",
        ["140 件已知真值样本", "仅用于优化", "test 集不参与"], C_INPUT)
    arrow(ax, (0.385, 0.17), (0.415, 0.17))

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print("saved:", OUT)


if __name__ == "__main__":
    main()

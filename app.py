# -*- coding: utf-8 -*-
"""
质检 demo 前端（Gradio）。在 AUP Learning Cloud 的 JupyterLab 里运行：

    python app.py

若通过 jupyter-server-proxy 访问，地址形如：
    https://tpe.aupcloud.io/user/<你的用户名>/proxy/7860/
"""
import json
import os
import random
import re

import cv2
import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qc_demo.synthetic_data import generate_dataset
from qc_demo.pipeline import (QCPipeline, prepare, evaluate_points, optimize_loop,
                              baseline_single_threshold)
from qc_demo.llm_client import LLMClient

# 中文字体（云上通常是 Noto Sans CJK）
for f in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]:
    try:
        matplotlib.rcParams["font.family"] = f
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

SEED = 2026
random.seed(SEED)
np.random.seed(SEED)

print("[demo] 初始化：生成合成数据 + 抽取感知特征 ...")
DEV_IMGS, DEV_LABELS = generate_dataset(120, seed=7)
TEST_IMGS, TEST_LABELS = generate_dataset(120, seed=99)
DEV_POINTS = prepare(DEV_IMGS, DEV_LABELS)
TEST_POINTS = prepare(TEST_IMGS, TEST_LABELS)

LLM = LLMClient()
MODE = f"真实 LLM（{LLM.model}）" if LLM.real else "内置确定性模拟器（未配置 API，可离线演示）"

STATE = {"pipe": None, "history": None, "opt_pipe": None}


# ------------------------------------------------------------------ 工具
def _rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def _pretty(d):
    return json.dumps(d, ensure_ascii=False, indent=2)


def fmt_metrics(m):
    return (f"漏检率 **{m['漏检率 FNR']:.1%}**　过杀率 **{m['过杀率 FPR']:.1%}**　"
            f"查全 **{m['查全率 Recall']:.1%}**　类型准确 **{m['类型准确率']:.1%}**　"
            f"分级一致 **{m['分级一致率']:.1%}**　"
            f"万元损失 **{m['万元损失(元/万件)']:,.0f} 元/万件**")


# ------------------------------------------------------------------ Tab1 单件抽检
def random_sample(part_choice):
    pipe = STATE["pipe"] or QCPipeline(LLM)
    STATE["pipe"] = pipe
    pool = [(im, lb) for im, lb in zip(TEST_IMGS, TEST_LABELS)
            if part_choice in ("随机", lb["part"])]
    img, lab = random.choice(pool)
    r = pipe.inspect(img, lab["part"], return_heatmap=True)
    heat = cv2.cvtColor(r["heatmap"], cv2.COLOR_BGR2RGB)

    gt = (f"工件：{lab['part']}　真值缺陷：{lab['defect']}　真值严重度：{lab['severity']}　"
          f"应按 SOP 处置：{lab['grade']}")
    pred_line = f"系统判定：{r['analysis'].get('defect_type')}/{r['analysis'].get('severity')} → {r['grade'].get('grade')}"
    hit = (r["analysis"].get("defect_type") == lab["defect"]) and (r["grade"].get("grade") == lab["grade"])
    verdict = "✅ 判定与真值一致" if hit else "❌ 判定与真值不一致"
    return (_rgb(img), heat, gt + "\n" + pred_line + "　" + verdict,
            _pretty(r["perception"]), _pretty(r["analysis"]), _pretty(r["grade"]))


# ------------------------------------------------------------------ Tab2 优化
def run_optimization(rounds):
    pipe, hist = optimize_loop(DEV_IMGS, DEV_LABELS, rounds=int(rounds),
                               llm=LLM, verbose=False, points=DEV_POINTS)
    STATE["opt_pipe"], STATE["history"] = pipe, hist

    xs = [h["round"] for h in hist]
    fnr = [h["metrics"]["漏检率 FNR"] for h in hist]
    fpr = [h["metrics"]["过杀率 FPR"] for h in hist]
    typ = [h["metrics"]["类型准确率"] for h in hist]
    grd = [h["metrics"]["分级一致率"] for h in hist]

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(xs, fnr, "o-", color="#c0392b", label="漏检率 FNR")
    ax[0].plot(xs, fpr, "s-", color="#e67e22", label="过杀率 FPR")
    ax[0].set_title("检测指标（越低越好）")
    ax[0].set_xlabel("优化轮次")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(xs, typ, "^-", color="#2c7fb8", label="缺陷类型准确率")
    ax[1].plot(xs, grd, "d-", color="#27ae60", label="分级一致率")
    ax[1].set_title("决策指标（越高越好）")
    ax[1].set_xlabel("优化轮次")
    ax[1].set_ylim(0, 1.02)
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()

    log_lines = []
    for h in hist:
        tag = "" if h.get("accepted", True) else "（回退）"
        log_lines.append(f"### Round {h['round']} {tag}")
        log_lines.append("")
        for a in h["actions"]:
            log_lines.append(f"- {a}")
        log_lines.append("")

    before = QCPipeline(LLM).analyzer.prompt
    after = pipe.analyzer.prompt
    m_before, _ = evaluate_points(QCPipeline(LLM), TEST_POINTS)
    m_after, _ = evaluate_points(pipe, TEST_POINTS)
    summary = (f"**优化前 @test**：{fmt_metrics(m_before)}\n\n"
               f"**优化后 @test**：{fmt_metrics(m_after)}\n\n"
               f"（优化只用了 dev 集 {len(DEV_LABELS)} 件样本，test 集 {len(TEST_LABELS)} 件完全留出）")
    return fig, "\n".join(log_lines), before, after, summary


# ------------------------------------------------------------------ Tab3 对照
def show_compare():
    m_before, _ = evaluate_points(QCPipeline(LLM), TEST_POINTS)
    pipe = STATE["opt_pipe"]
    if pipe is None:
        return "请先在「提示词自动优化」页签运行一次优化，再回来看对照实验。"
    m_after, _ = evaluate_points(pipe, TEST_POINTS)
    b = baseline_single_threshold(DEV_POINTS, TEST_POINTS)
    md = f"""
| 方案 | 漏检率 | 过杀率 | 查全率 | 类型准确率 | 分级一致率 | 万元损失 |
|---|---|---|---|---|---|---|
| 基线1：单模型·异常分阈值（阈值={b['阈值']}） | {b['漏检率 FNR']:.1%} | {b['过杀率 FPR']:.1%} | {b['查全率 Recall']:.1%} | 不支持 | 不支持 | {b['万元损失(元/万件)']:,.0f} |
| 基线2：多智能体·人工经验提示词 | {m_before['漏检率 FNR']:.1%} | {m_before['过杀率 FPR']:.1%} | {m_before['查全率 Recall']:.1%} | {m_before['类型准确率']:.1%} | {m_before['分级一致率']:.1%} | {m_before['万元损失(元/万件)']:,.0f} |
| 本方案：多智能体·提示词自动优化 | {m_after['漏检率 FNR']:.1%} | {m_after['过杀率 FPR']:.1%} | {m_after['查全率 Recall']:.1%} | {m_after['类型准确率']:.1%} | {m_after['分级一致率']:.1%} | {m_after['万元损失(元/万件)']:,.0f} |

**读法**：基线1 只能回答「良/不良」，且为压低漏检必须把阈值降到 12% 过杀；多智能体在 0% 过杀下把漏检压到 0，
并额外给出缺陷类型与处置建议——这是单模型做不到的部分。
"""
    return md


with gr.Blocks(title="多智能体工业质检 Demo") as demo:
    gr.Markdown("# 🔧 多智能体 + 提示词自动优化 · 螺丝/螺母质检 Demo")
    gr.Markdown(f"运行模式：**{MODE}**　|　架构：感知层(经典CV) → 缺陷分析 Agent → 分级决策 Agent ⇄ 反思优化 Agent")

    with gr.Tab("① 单件抽检"):
        with gr.Row():
            part = gr.Radio(["随机", "screw", "nut"], value="随机", label="工件类型")
            btn = gr.Button("🎲 抽检一件", variant="primary", scale=0)
        with gr.Row():
            with gr.Column():
                img_out = gr.Image(label="原图", height=260)
                gt_out = gr.Textbox(label="真值 vs 判定", lines=3)
            with gr.Column():
                heat_out = gr.Image(label="异常热力图（感知层）", height=260)
                perc_out = gr.Code(label="① 感知报告", language="json")
        with gr.Row():
            ana_out = gr.Code(label="② 缺陷分析 Agent 输出", language="json")
            grd_out = gr.Code(label="③ 分级决策 Agent 输出", language="json")
        btn.click(random_sample, inputs=[part], outputs=[img_out, heat_out, gt_out, perc_out, ana_out, grd_out])

    with gr.Tab("② 提示词自动优化"):
        gr.Markdown("反思 Agent 会从误判中诊断「哪条判据错了」，再用 dev 集的标注样本反推新的判据边界，写回提示词。")
        with gr.Row():
            rounds = gr.Slider(4, 12, value=10, step=1, label="优化轮数")
            run_btn = gr.Button("▶ 运行优化闭环", variant="primary")
        plot_out = gr.Plot(label="优化曲线")
        summary_out = gr.Markdown()
        log_out = gr.Markdown(label="每轮反思记录")
        with gr.Row():
            before_out = gr.Textbox(label="优化前提示词", lines=22)
            after_out = gr.Textbox(label="优化后提示词", lines=22)
        run_btn.click(run_optimization, inputs=[rounds],
                      outputs=[plot_out, log_out, before_out, after_out, summary_out])

    with gr.Tab("③ 对照实验"):
        gr.Markdown("对照：单模型阈值 vs 多智能体（人工提示词）vs 多智能体（自动优化）")
        cmp_btn = gr.Button("📊 生成对照表", variant="primary")
        cmp_out = gr.Markdown()
        cmp_btn.click(show_compare, outputs=[cmp_out])

    gr.Markdown("---\n数据为程序化合成的螺丝/螺母缺陷图像（含划痕、变形/色差、缺料、锈蚀，含机加工纹理与光照波动），"
                "每件自带真值标注。替换为 MVTec AD 真实数据的方法见 README。")


def _detect_root_path(port: int) -> str:
    """推断反向代理前缀（子路径），用于告诉 Gradio「我挂在哪个子路径下」。

    云端（JupyterHub + jupyter-server-proxy / code-server 端口转发）会把本服务挂在
    `/user/<用户名>/proxy/<端口>` 这样的子路径下。代理转发时会把前缀剥掉，而 Gradio
    前端默认认为自己就在域名根目录，于是用绝对路径 `/gradio_api/...` 发请求 —— 该请求
    被平台根路径的 HTML 页面接走，前端解析 JSON 时就报：
        Could not parse server response: SyntaxError: Unexpected token '<'
    （`<` 就是 HTML 的开头）。

    最稳的写法是给**完整 URL**（Gradio 会原样作为 root 使用）：
        QCDEMO_ROOT_PATH=https://tpe.aupcloud.io/user/<用户名>/proxy/7860 python app.py
    也接受纯子路径（此时 Gradio 会用请求的 origin 拼前面）：
        QCDEMO_ROOT_PATH=/user/<用户名>/proxy/7860 python app.py
    """
    explicit = (os.environ.get("QCDEMO_ROOT_PATH")
                or os.environ.get("GRADIO_ROOT_PATH") or "").strip()
    if explicit:
        return explicit.rstrip("/")

    # 在 JupyterHub 派生的终端里，这个环境变量天然存在，形如 /user/bjut_teach_077/
    prefix = (os.environ.get("JUPYTERHUB_SERVICE_PREFIX") or "").strip()
    if prefix:
        path = f"{prefix.rstrip('/')}/proxy/{port}"
        domain = (os.environ.get("QCDEMO_DOMAIN") or "").strip().rstrip("/")
        return f"{domain}{path}" if domain else path
    return ""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    root_path = _detect_root_path(port)
    kwargs = dict(server_name="0.0.0.0", server_port=port, theme=gr.themes.Soft())
    if root_path:
        kwargs["root_path"] = root_path
        print(f"[i] root_path = {root_path}")
        print(f"[i] 访问地址：{root_path if root_path.startswith('http') else 'https://<平台域名>' + root_path}/")
        print("[i] 若页面能打开但报 Unexpected token '<'，说明前缀不对，请用完整 URL 形式重启：")
        print(f"    QCDEMO_ROOT_PATH=https://<平台域名>/user/<用户名>/proxy/{port} python app.py")
    else:
        print(f"[i] 未检测到代理前缀，按根路径启动：http://127.0.0.1:{port}/")
        print("[i] 若在云端通过 proxy 子路径访问，请显式指定完整 URL 前缀：")
        print(f"    QCDEMO_ROOT_PATH=https://<平台域名>/user/<用户名>/proxy/{port} python app.py")
    demo.launch(**kwargs)

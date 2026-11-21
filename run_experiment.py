# -*- coding: utf-8 -*-
"""
完整实验脚本（无 UI 依赖）：跑一遍对照实验 + 提示词优化闭环，打印可直接抄进报告的表格。

用法：
    python run_experiment.py
"""
import json

from qc_demo.synthetic_data import generate_dataset
from qc_demo.pipeline import (QCPipeline, prepare, evaluate_points, optimize_loop,
                              baseline_single_threshold)
from qc_demo.llm_client import LLMClient


def fmt(m):
    return (f"漏检={m['漏检率 FNR']:.3f} 过杀={m['过杀率 FPR']:.3f} "
            f"查全={m['查全率 Recall']:.3f} 类型={m['类型准确率']:.3f} "
            f"分级={m['分级一致率']:.3f} 综合={m['综合准确率']:.3f} "
            f"损失={m['万元损失(元/万件)']:.0f}")


if __name__ == "__main__":
    print("=" * 78)
    llm = LLMClient()
    print("LLM 模式:", f"真实 API ({llm.model})" if llm.real else "内置确定性模拟器")

    dev_imgs, dev_labels = generate_dataset(140, seed=7)
    test_imgs, test_labels = generate_dataset(160, seed=99)
    print(f"dev={len(dev_labels)} 件, test={len(test_labels)} 件")

    dev_points = prepare(dev_imgs, dev_labels)
    test_points = prepare(test_imgs, test_labels)

    # ---------- 基线 1：单阈值二分类 ----------
    b1 = baseline_single_threshold(dev_points, test_points)
    print("\n【基线1】单模型·异常分数阈值（dev 选阈值，test 汇报；无类型、无分级）")
    print("   ", b1)
    print(f"    → 若强制零过杀（阈值={b1['零过杀阈值']}），单阈值漏检率高达 {b1['零过杀下漏检率']:.1%}")

    # ---------- 基线 2：多智能体 + 工程师经验提示词（未优化） ----------
    pipe0 = QCPipeline(llm)
    m0, _ = evaluate_points(pipe0, test_points)
    print("\n【基线2】多智能体班组·人工经验提示词（未优化，@test）")
    print("   ", fmt(m0))

    # ---------- 方法：dev 集上自动优化提示词 ----------
    print("\n【方法】提示词自动优化（只在 dev 集上优化）")
    pipe1, history = optimize_loop(dev_imgs, dev_labels, rounds=10, verbose=True, points=dev_points)

    m1, _ = evaluate_points(pipe1, test_points)
    print("\n【结果】优化后 @test（独立测试集）")
    print("   ", fmt(m1))

    # ---------- 提升幅度 ----------
    print("\n" + "=" * 78)
    print("提升幅度（相对未优化）:")
    for k in ("漏检率 FNR", "过杀率 FPR", "类型准确率", "分级一致率", "综合准确率"):
        print(f"  {k}: {m0[k]:.3f} -> {m1[k]:.3f}  ({m1[k] - m0[k]:+.3f})")
    print(f"  万元损失: {m0['万元损失(元/万件)']:.0f} -> {m1['万元损失(元/万件)']:.0f} 元/万件")

    # ---------- 提示词演化 ----------
    print("\n优化过程中的判据变化:")
    for exp in getattr(pipe1, "experience", []):
        print(f"  Round {exp['round']}: 误判 {exp['n_errors']} 例 "
              f"(漏检{exp['miss']}/过杀{exp['overkill']}/类型{exp['type_err']}/分级{exp['grade_err']})")
        for a in exp["actions"]:
            print(f"    · {a}")

    # ---------- 最终提示词 ----------
    with open("optimized_prompt.txt", "w", encoding="utf-8") as f:
        f.write(pipe1.analyzer.prompt)
    print("\n最终提示词已保存到 optimized_prompt.txt")
    print("\n最终判据表:")
    import re
    m = re.search(r"RULES:(.*?)END RULES", pipe1.analyzer.prompt, re.S)
    print(m.group(1).strip() if m else "(未找到)")

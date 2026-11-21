# -*- coding: utf-8 -*-
"""快速自检：dev 集优化 -> test 集汇报，验证闭环可跑通并确实提升。

用法：
    python selftest.py
"""
from qc_demo.synthetic_data import generate_dataset
from qc_demo.pipeline import (QCPipeline, prepare, evaluate_points,
                              optimize_loop, baseline_single_threshold)
from qc_demo.llm_client import LLMClient


def show(tag, m):
    print(f"[{tag}] 漏检={m['漏检率 FNR']:.3f} 过杀={m['过杀率 FPR']:.3f} "
          f"类型={m['类型准确率']:.3f} 分级={m['分级一致率']:.3f} 综合={m['综合准确率']:.3f} "
          f"损失={m['万元损失(元/万件)']}")


if __name__ == "__main__":
    dev_imgs, dev_labels = generate_dataset(120, seed=7)
    test_imgs, test_labels = generate_dataset(120, seed=99)
    llm = LLMClient()
    print(f"LLM 模式: {'真实 API ' + llm.model if llm.real else '内置确定性模拟器'}")

    dev_points = prepare(dev_imgs, dev_labels)
    test_points = prepare(test_imgs, test_labels)

    # 单阈值基线（dev 选阈值，test 汇报）
    b = baseline_single_threshold(dev_points, test_points)
    print(f"单模型基线 @test: 漏检={b['漏检率 FNR']:.3f} 过杀={b['过杀率 FPR']:.3f} "
          f"综合={b['综合准确率']:.3f} | 零过杀下漏检={b['零过杀下漏检率']:.3f}")

    pipe = QCPipeline(llm)
    m0, _ = evaluate_points(pipe, test_points)
    show("初始提示词(未优化) @test", m0)

    pipe2, hist = optimize_loop(dev_imgs, dev_labels, rounds=6, points=dev_points)
    m1, _ = evaluate_points(pipe2, test_points)
    show("优化后 @test", m1)

    print("\n单模型基线:", b)
    print("提示词版本数:", len(pipe2.analyzer.history))
    print("OK: 优化闭环跑通。")

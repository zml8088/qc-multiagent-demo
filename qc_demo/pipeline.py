# -*- coding: utf-8 -*-
"""
流水线 + 评测 + 提示词自动优化闭环

实验设计（写报告可直接引用）：
1. 感知特征与提示词无关，先一次性算好并缓存 —— 优化循环里只跑 Agent，速度快几十倍。
2. 提示词只在 dev 集上优化，最终指标在独立 test 集汇报，避免对着测试集调提示词。
3. 每轮改写后复评 dev，指标不变差才接受（keep-best），保证优化曲线单调不倒退。
4. 指标覆盖检测（漏检/过杀）与决策（类型/分级），并折算成产线口径的万元损失。
"""
from .llm_client import LLMClient
from .perception import PerceptionModule
from .agents import AnalyzerAgent, GraderAgent, ReflectorAgent, SEV2GRADE

COST_MISS = 500.0      # 漏检一件的成本（客户索赔/召回，元）
COST_OVERKILL = 80.0   # 过杀一件的成本（好件被误判报废/返修，元）


class QCPipeline:
    """单件质检：感知 → 分析 → 分级"""

    def __init__(self, llm=None):
        self.llm = llm or LLMClient()
        self.perception = PerceptionModule()
        self.analyzer = AnalyzerAgent(self.llm)
        self.grader = GraderAgent(self.llm)

    def inspect(self, img, part, return_heatmap=False):
        perc = self.perception.analyze(img, part)
        analysis = self.analyzer.run(perc)
        grade = self.grader.run(analysis)
        out = {"perception": perc, "analysis": analysis, "grade": grade}
        if return_heatmap:
            out["heatmap"] = self.perception.heatmap(img, part)
        return out

    def inspect_features(self, feats):
        analysis = self.analyzer.run(feats)
        grade = self.grader.run(analysis)
        return {"perception": feats, "analysis": analysis, "grade": grade}


# ---------------------------------------------------------------- 特征缓存
def prepare(images, labels):
    """一次性抽取感知特征，后续优化循环复用（感知层与提示词无关）"""
    perc = PerceptionModule()
    return [{"features": perc.analyze(img, lab["part"]), "label": lab}
            for img, lab in zip(images, labels)]


# ---------------------------------------------------------------- 评测
def evaluate_points(pipeline, points):
    tp = fp = fn = tn = 0
    type_hits = type_total = 0
    grade_hits = grade_total = 0
    errors = []

    for p in points:
        feats, lab = p["features"], p["label"]
        r = pipeline.inspect_features(feats)
        pred_type = r["analysis"].get("defect_type", "none")
        gt_def = lab["defect"] != "none"
        pr_def = pred_type != "none"

        if gt_def and pr_def:
            tp += 1
            type_total += 1
            if pred_type == lab["defect"]:
                type_hits += 1
            else:
                errors.append({"gt": lab, "pred": r["analysis"], "features": feats, "kind": "type"})
            grade_total += 1
            if r["grade"].get("grade") == lab["grade"]:
                grade_hits += 1
            else:
                errors.append({"gt": lab, "pred": r["analysis"], "features": feats, "kind": "grade"})
        elif gt_def and not pr_def:
            fn += 1
            errors.append({"gt": lab, "pred": r["analysis"], "features": feats, "kind": "miss"})
        elif not gt_def and pr_def:
            fp += 1
            errors.append({"gt": lab, "pred": r["analysis"], "features": feats, "kind": "overkill"})
        else:
            tn += 1

    n_def, n_ok, n_all = tp + fn, fp + tn, tp + tn + fp + fn
    metrics = {
        "漏检率 FNR": round(fn / n_def, 4) if n_def else 0.0,
        "过杀率 FPR": round(fp / n_ok, 4) if n_ok else 0.0,
        "查全率 Recall": round(tp / n_def, 4) if n_def else 1.0,
        "查准率 Precision": round(tp / (tp + fp), 4) if (tp + fp) else 1.0,
        "类型准确率": round(type_hits / type_total, 4) if type_total else 1.0,
        "分级一致率": round(grade_hits / grade_total, 4) if grade_total else 1.0,
        "综合准确率": round((tp + tn) / max(1, n_all), 4),
        "万元损失(元/万件)": round((fn * COST_MISS + fp * COST_OVERKILL) / max(1, n_all) * 10000, 1),
        "样本数": n_all, "缺陷数": n_def, "良品数": n_ok,
    }
    return metrics, errors


def evaluate(pipeline, images, labels):
    return evaluate_points(pipeline, prepare(images, labels))


# ---------------------------------------------------------------- 优化闭环
def optimize_loop(dev_images, dev_labels, rounds=10, llm=None, verbose=True, points=None):
    """在 dev 集上迭代校准提示词；返回 (pipeline, history)"""
    pipeline = QCPipeline(llm)
    reflector = ReflectorAgent(pipeline.llm, pipeline.analyzer)
    points = points or prepare(dev_images, dev_labels)
    history = []

    best, errors = evaluate_points(pipeline, points)
    best_prompt = pipeline.analyzer.prompt
    history.append({"round": 0, "metrics": best, "actions": ["初始提示词（工程师凭经验设定的判据）"]})
    if verbose:
        _print_round(0, best, ["初始提示词（工程师凭经验设定的判据）"])

    for r in range(1, rounds + 1):
        new_prompt = reflector.reflect(errors, pipeline.analyzer.prompt, r, points)
        pipeline.analyzer.prompt = new_prompt
        pipeline.analyzer.history.append(new_prompt)
        metrics, new_errors = evaluate_points(pipeline, points)

        accepted = _score(metrics) >= _score(best)
        if accepted:
            best, best_prompt, errors = metrics, new_prompt, new_errors
        else:  # 回退到当前最优提示词
            pipeline.analyzer.prompt = best_prompt
            pipeline.analyzer.history[-1] = best_prompt
            metrics, new_errors = best, errors

        actions = reflector.experience[-1]["actions"]
        history.append({"round": r, "metrics": metrics, "actions": actions, "accepted": accepted})
        if verbose:
            _print_round(r, metrics, actions, accepted)

        if metrics["漏检率 FNR"] == 0 and metrics["过杀率 FPR"] == 0 \
                and metrics["类型准确率"] >= 0.99 and metrics["分级一致率"] >= 0.99:
            if verbose:
                print(">>> 已达目标水平，提前收敛。")
            break

    pipeline.experience = reflector.experience
    pipeline.prompt_versions = reflector.prompt_versions
    return pipeline, history


def _score(m):
    """综合目标函数：漏检权重最高（工业上漏检＝不良品流出，代价最大）"""
    return -(m["漏检率 FNR"] * 3 + m["过杀率 FPR"] * 1.0 +
             (1 - m["类型准确率"]) * 1.0 + (1 - m["分级一致率"]) * 0.6)


def _print_round(r, m, actions, accepted=True):
    tag = "接受" if accepted else "拒绝(回退)"
    print(f"--- Round {r} [{tag}] " + "-" * 34)
    print(f"    漏检={m['漏检率 FNR']:.3f} 过杀={m['过杀率 FPR']:.3f} "
          f"类型={m['类型准确率']:.3f} 分级={m['分级一致率']:.3f} "
          f"综合={m['综合准确率']:.3f} 损失={m['万元损失(元/万件)']:.0f}元/万件")
    for a in actions:
        print(f"    · {a}")


# ---------------------------------------------------------------- 对照基线
def baseline_single_threshold(tune_points, eval_points=None):
    """单模型基线：只用异常分数做良/不良二分类（无类型、无分级）。

    阈值在 tune_points（dev）上选，指标在 eval_points（test）上汇报，与主方法口径一致。
    """
    eval_points = eval_points or tune_points

    def metrics_at(points, th):
        tp = fp = fn = tn = 0
        for p in points:
            s = p["features"]["anomaly_score_pct"]
            gt = p["label"]["defect"] != "none"
            pred = s >= th
            if gt and pred:
                tp += 1
            elif gt and not pred:
                fn += 1
            elif not gt and pred:
                fp += 1
            else:
                tn += 1
        n_def, n_ok, n_all = tp + fn, fp + tn, tp + tn + fp + fn
        return {
            "漏检率 FNR": round(fn / n_def, 4) if n_def else 0.0,
            "过杀率 FPR": round(fp / n_ok, 4) if n_ok else 0.0,
            "查全率 Recall": round(tp / n_def, 4) if n_def else 1.0,
            "综合准确率": round((tp + tn) / max(1, n_all), 4),
            "万元损失(元/万件)": round((fn * COST_MISS + fp * COST_OVERKILL) / max(1, n_all) * 10000, 1),
            "类型准确率": "不支持", "分级一致率": "不支持",
        }

    # 在 dev 上按综合准确率选阈值（并列时取查全率高者）
    best_th, best_key = 1.0, None
    for i in range(1, 400):
        th = i / 100
        m = metrics_at(tune_points, th)
        key = (m["综合准确率"], m["查全率 Recall"])
        if best_key is None or key > best_key:
            best_th, best_key = th, key

    out = metrics_at(eval_points, best_th)
    out["阈值"] = best_th

    # 附加口径：强制零过杀（产线常见硬约束）时单阈值能达到的漏检率
    good_scores = [p["features"]["anomaly_score_pct"] for p in tune_points
                   if p["label"]["defect"] == "none"]
    th0 = (max(good_scores) if good_scores else 0) + 0.01
    m0 = metrics_at(eval_points, th0)
    out["零过杀阈值"] = round(th0, 2)
    out["零过杀下漏检率"] = m0["漏检率 FNR"]
    return out

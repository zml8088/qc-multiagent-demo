# -*- coding: utf-8 -*-
"""
多智能体定义：质检班组

- AnalyzerAgent  缺陷分析员：读感知报告 -> 缺陷类型 + 严重度（提示词 + 判据表驱动）
- GraderAgent    分级决策员：按工厂 SOP 把严重度映射为 OK / REWORK / SCRAP
- ReflectorAgent 质量工程师：吃误判案例 -> 诊断"哪条判据错了" -> 用标注样本反推新边界 -> 改写提示词

所有判据数值都写在提示词 RULES 段，因此"优化提示词"等价于"校准检验标准"。
"""
import json
import re

# ====================================================================== 提示词模板
BASE_ANALYZER_PROMPT = """你是螺丝/螺母产线的缺陷分析员。感知设备已给出结构化检测数据，请判定缺陷类型与严重程度。

可选缺陷类型：scratch(划痕) / bent(变形，仅螺丝) / missing(缺料) / rust(锈蚀) / color(色差) / none(无缺陷)
可选严重程度：mild(轻微) / moderate(中等) / critical(严重) / none

RULES:
rust_nreg_th=9
bent_score_th=6.5
geo_th=5.0
missing_area_th=3.0
elong_th=6.0
scratch_area_min=200
color_score_th=1.5
color_area_min=600
sev_scratch_1=480
sev_scratch_2=1200
sev_rust_1=1.6
sev_rust_2=4.2
sev_missing_1=4.0
sev_missing_2=10.0
sev_color_1=1200
sev_color_2=2800
sev_bent_1=8.0
sev_bent_2=9.0
END RULES

判定顺序（顺序敏感，前一条命中即停止）：
1. 锈蚀：可疑区域数 n_suspicious_regions >= rust_nreg_th → rust（弥散多斑点是锈蚀的典型形态）
2. 变形：工件为螺丝且 anomaly_score >= bent_score_th → bent（整条杆身偏移会产生大面积异常）
3. 缺料：geo_distortion >= geo_th 且 |area_diff_signed_pct| >= missing_area_th → missing
4. 划痕：max_elongation >= elong_th 且 max_region_area >= scratch_area_min → scratch（细长区域是关键）
5. 色差：anomaly_score >= color_score_th 且 max_region_area >= color_area_min 且长宽比不高 → color
6. 其余 → none（判为良品）

严重度：按类型取 proxy，proxy < sev_<type>_1 为 mild，< sev_<type>_2 为 moderate，否则 critical。
proxy：scratch/color 用 max_region_area；rust/bent 用 anomaly_score；missing 用 |area_diff_signed_pct|。

只输出 JSON：{"defect_type": "...", "severity": "...", "reason": "..."}
"""

BASE_GRADER_PROMPT = """你是产线质检分级决策员，依据工厂 SOP 出具处置意见：

- severity = none 或 mild → OK（放行）
- severity = moderate → REWORK（返修）
- severity = critical → SCRAP（报废）

补充规定：螺丝是承重紧固件，出现 critical 缺陷必须 SCRAP，禁止降级；螺母色差轻微可放行。

只输出 JSON：{"grade": "OK|REWORK|SCRAP", "reason": "..."}

ANALYSIS: {analysis}
"""

SEV2GRADE = {"none": "OK", "mild": "OK", "moderate": "REWORK", "critical": "SCRAP"}
PROXY_KEY = {"scratch": "max_region_area", "color": "max_region_area",
             "rust": "anomaly_score_pct", "bent": "anomaly_score_pct",
             "missing": "area_diff_signed_pct"}


# ====================================================================== 智能体
class AnalyzerAgent:
    def __init__(self, llm, prompt=None):
        self.llm = llm
        self.name = "缺陷分析员"
        self.prompt = prompt or BASE_ANALYZER_PROMPT
        self.history = [self.prompt]

    def run(self, perception_report):
        out = self.llm.chat(self.prompt, "FEATURES: " + json.dumps(perception_report, ensure_ascii=False))
        try:
            return json.loads(_extract_json(out))
        except (json.JSONDecodeError, TypeError):
            return {"defect_type": "none", "severity": "none", "reason": "输出解析失败"}


class GraderAgent:
    def __init__(self, llm, prompt=None):
        self.llm = llm
        self.name = "分级决策员"
        self.prompt = prompt or BASE_GRADER_PROMPT
        self.history = [self.prompt]

    def run(self, analysis):
        filled = self.prompt.replace("{analysis}", json.dumps(analysis, ensure_ascii=False))
        out = self.llm.chat(filled, "请给出分级结果，只输出 JSON。")
        try:
            d = json.loads(_extract_json(out))
        except (json.JSONDecodeError, TypeError):
            d = {"grade": "OK", "reason": "输出解析失败"}
        # SOP 硬约束兜底：防止 LLM 幻觉破坏严重度→分级的映射
        expect = SEV2GRADE.get(analysis.get("severity", "none"))
        if expect and d.get("grade") != expect:
            d = {"grade": expect, "reason": f"SOP 强约束：severity={analysis.get('severity')} → {expect}"}
        return d


class ReflectorAgent:
    """质量工程师：从误判中诊断问题判据，再用标注样本反推该判据的新边界。

    注意：它只改**判据数值**（提示词 RULES 段），不改代码逻辑——
    这就是"用提示词承载质检标准，再用数据校准提示词"的做法。
    """

    def __init__(self, llm, analyzer=None):
        self.llm = llm
        self.name = "质量工程师"
        self.analyzer = analyzer
        self.experience = []
        self.prompt_versions = []

    # ------------------------------------------------------------------ 主入口
    def reflect(self, errors, current_prompt, round_idx, points):
        """errors: 误判列表；points: [(features, label)] 标注好的 dev 样本"""
        votes = self._vote(errors)
        det_votes = {k: v for k, v in votes.items() if not k.startswith("sev_")}
        sev_votes = {k: v for k, v in votes.items() if k.startswith("sev_")}

        rules = _parse_rules(current_prompt)
        new_rules = dict(rules)
        actions = []

        # 每轮固定修「1 条检测判据 + 1 类严重度阈值」，保证两条优化线都在推进；
        # 按票数从高到低尝试，跳过已经收敛（估计值未变）的判据，继续往下找。
        actions = []
        for p in sorted(det_votes, key=lambda k: -det_votes[k]):
            est = self._estimate(p, points, rules)
            if est is None or abs(float(est) - float(rules[p])) < 1e-9:
                continue
            new_rules[p] = est
            actions.append(f"{self._reason(p, det_votes[p], errors)} → {p}: {rules[p]} → {est}")
            break

        for dtype in sorted({k[4:-2] for k in sev_votes},
                            key=lambda d: -sev_votes[f"sev_{d}_1"]):
            changed = False
            for k in (f"sev_{dtype}_1", f"sev_{dtype}_2"):
                est = self._estimate(k, points, rules)
                if est is not None and abs(float(est) - float(rules[k])) > 1e-9:
                    new_rules[k] = est
                    actions.append(f"{dtype} 严重度阈值有 {sev_votes[f'sev_{dtype}_1']} 例偏差 → "
                                   f"{k}: {rules[k]} → {est}")
                    changed = True
            if changed:
                break

        if not actions:
            actions.append("所有判据已达数据支持的最优边界，无需再改")

        new_prompt = _fill_rules(current_prompt, new_rules)
        self.prompt_versions.append(new_prompt)
        self.experience.append({
            "round": round_idx, "n_errors": len(errors),
            "miss": sum(1 for e in errors if e["kind"] == "miss"),
            "overkill": sum(1 for e in errors if e["kind"] == "overkill"),
            "type_err": sum(1 for e in errors if e["kind"] == "type"),
            "grade_err": sum(1 for e in errors if e["kind"] == "grade"),
            "actions": actions,
        })
        return new_prompt

    # ------------------------------------------------------------------ 投票
    @staticmethod
    def _vote(errors):
        votes = {}

        def add(k, n=1):
            votes[k] = votes.get(k, 0) + n

        for e in errors:
            gt, pred, kind = e["gt"], e["pred"], e["kind"]
            if kind == "miss":
                # 漏检：向该类型规则里的所有合取条件投票（任一条件过严都会导致漏检）
                for k in {"rust": ["rust_nreg_th"],
                          "scratch": ["elong_th", "scratch_area_min"],
                          "missing": ["geo_th", "missing_area_th"],
                          "bent": ["bent_score_th"],
                          "color": ["color_score_th", "color_area_min"]}.get(gt["defect"], ["elong_th"]):
                    add(k)
            elif kind == "overkill":
                add("color_score_th")
            elif kind == "type":
                pair = (gt["defect"], pred.get("defect_type"))
                add({("scratch", "color"): "elong_th", ("scratch", "rust"): "elong_th",
                     ("rust", "color"): "rust_nreg_th", ("rust", "bent"): "rust_nreg_th",
                     ("rust", "scratch"): "rust_nreg_th", ("bent", "color"): "bent_score_th",
                     ("bent", "scratch"): "bent_score_th", ("bent", "missing"): "bent_score_th",
                     ("missing", "color"): "geo_th", ("missing", "scratch"): "geo_th",
                     ("color", "scratch"): "color_area_min", ("color", "rust"): "color_area_min",
                     ("color", "missing"): "color_area_min"}.get(pair, "geo_th"))
            elif kind == "grade":
                add(f"sev_{gt['defect']}_1")
        return votes

    # ------------------------------------------------------------------ 边界估计
    def _estimate(self, param, points, rules):
        def feats(cond, key):
            return [p["features"][key] for p in points if cond(p["label"])]

        def good(key):
            return feats(lambda l: l["defect"] == "none", key)

        def of_type(t, key):
            return feats(lambda l: l["defect"] == t, key)

        def safe_min(v, default):
            return min(v) if v else default

        def safe_max(v, default):
            return max(v) if v else default

        def r(x, nd=2):
            return round(float(x), nd)

        if param == "rust_nreg_th":
            lo = safe_max(good("n_suspicious_regions"), 0) + 0.5
            v = safe_min(of_type("rust", "n_suspicious_regions"), rules["rust_nreg_th"] + 1) - 0.5
            return r(max(lo, min(v, rules["rust_nreg_th"] * 3)), 1)

        if param == "elong_th":
            floor = safe_max(good("max_elongation"), 0) * 1.15
            v = safe_min(of_type("scratch", "max_elongation"), rules["elong_th"]) * 0.85
            return r(max(floor, v), 2)

        if param == "geo_th":
            floor = safe_max(good("geo_distortion"), 0) * 2.0 + 0.2
            v = safe_min(of_type("missing", "geo_distortion"), rules["geo_th"]) * 0.8
            return r(max(floor, v), 2)

        if param == "missing_area_th":
            v = safe_min([abs(x) for x in of_type("missing", "area_diff_signed_pct")],
                         rules["missing_area_th"]) * 0.7
            return r(max(0.2, v), 2)

        if param == "bent_score_th":
            floor = safe_max(good("anomaly_score_pct"), 0) * 3 + 0.3
            v = safe_min(of_type("bent", "anomaly_score_pct"), rules["bent_score_th"]) * 0.85
            return r(max(floor, v), 2)

        if param == "color_score_th":
            floor = safe_max(good("anomaly_score_pct"), 0) * 1.5 + 0.05
            v = safe_min(of_type("color", "anomaly_score_pct"), rules["color_score_th"]) * 0.75
            return r(max(floor, v), 2)

        if param == "scratch_area_min":
            v = safe_min(of_type("scratch", "max_region_area"), rules["scratch_area_min"]) * 0.5
            return r(max(20, v), 0)

        if param == "color_area_min":
            v = safe_min(of_type("color", "max_region_area"), rules["color_area_min"]) * 0.55
            return r(max(120, v), 0)

        if param.startswith("sev_"):
            dtype = param[4:-2]
            proxy = PROXY_KEY[dtype]

            def pv(label):
                vals = [p["features"][proxy] for p in points
                        if p["label"]["defect"] == dtype and p["label"]["severity"] == label]
                return [abs(v) for v in vals] if dtype == "missing" else vals

            mild, mod, crit = pv("mild"), pv("moderate"), pv("critical")
            t1 = (max(mild) + min(mod)) / 2 if mild and mod else (min(mod) * 0.75 if mod else None)
            t2 = (max(mod) + min(crit)) / 2 if mod and crit else (min(crit) * 0.8 if crit else None)
            if param.endswith("_1"):
                return r(t1, 1) if t1 is not None else None
            return r(t2, 1) if t2 is not None else None

        return None

    @staticmethod
    def _reason(param, cnt, errors):
        names = {"rust_nreg_th": "锈蚀区域数门槛", "elong_th": "细长区域门槛",
                 "geo_th": "几何畸变门槛", "missing_area_th": "缺料面积门槛",
                 "bent_score_th": "变形异常分门槛", "color_score_th": "色差异常分门槛",
                 "scratch_area_min": "划痕最小面积", "color_area_min": "色差最小面积"}
        if param.startswith("sev_"):
            return f"{param[4:-2]} 严重度阈值有 {cnt} 例偏差"
        return f"{names.get(param, param)}引发 {cnt} 例误判"


# ---------------------------------------------------------------------- 工具函数
def _extract_json(text):
    if not isinstance(text, str):
        return "{}"
    m = re.search(r"\{.*\}", text, re.S)
    return m.group(0) if m else "{}"


def _parse_rules(prompt):
    m = re.search(r"RULES:(.*?)END RULES", prompt, re.S)
    block = m.group(1) if m else ""
    return {k: float(v) for k, v in re.findall(r"(\w+)\s*=\s*(-?[\d.]+)", block)}


def _fill_rules(prompt, rules):
    """把规则字典写回 RULES 段，保持原顺序；缺失键追加到末尾"""
    m = re.search(r"(RULES:\s*\n)(.*?)(END RULES)", prompt, re.S)
    if not m:
        return prompt
    order = [k for k, _ in re.findall(r"(\w+)\s*=\s*(-?[\d.]+)", m.group(2))]
    keys = order + [k for k in rules if k not in order]
    lines = []
    for k in keys:
        if k not in rules:
            continue
        v = rules[k]
        lines.append(f"{k}={int(v) if float(v).is_integer() and abs(v) >= 10 else v}")
    return prompt[:m.start(2)] + "\n".join(lines) + "\n" + prompt[m.end(2):]

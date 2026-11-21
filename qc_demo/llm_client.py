# -*- coding: utf-8 -*-
"""
LLM 客户端：OpenAI 兼容 API 的可插拔封装。

- 配了 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 就调真实 LLM（AUP 云端 vLLM / Ollama / 任意兼容服务）
- 否则用内置 DeterministicLLM：从提示词 RULES 段读判据参数，按同一套规则树判定。
  这样"改提示词"真的会改变输出，提示词优化闭环在无 API 时依然成立、可复现。
"""
import json
import os
import re

DEFAULT_RULES = {
    # 判据树阈值
    "rust_nreg_th": 9, "bent_score_th": 6.5, "geo_th": 5.0,
    "missing_area_th": 3.0, "elong_th": 6.0,
    "scratch_area_min": 200, "color_score_th": 1.5, "color_area_min": 600,
    # 严重度阈值表（按类型）
    "sev_scratch_1": 480, "sev_scratch_2": 1200,
    "sev_rust_1": 1.6, "sev_rust_2": 4.2,
    "sev_missing_1": 4.0, "sev_missing_2": 10.0,
    "sev_color_1": 1200, "sev_color_2": 2800,
    "sev_bent_1": 8.0, "sev_bent_2": 9.0,
}


class LLMClient:
    def __init__(self, base_url=None, api_key=None, model=None):
        self.base_url = base_url or os.environ.get("LLM_BASE_URL")
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        self.model = model or os.environ.get("LLM_MODEL", "qwen2.5-vl-7b-instruct")
        self.real = bool(self.base_url and self.api_key)
        self.call_log = []

    def chat(self, system_prompt, user_prompt):
        self.call_log.append({"system": system_prompt, "user": user_prompt})
        if self.real:
            return self._call_real(system_prompt, user_prompt)
        return self._simulate(system_prompt + "\n" + user_prompt)

    # ------------------------------------------------------------------ 真实 API
    def _call_real(self, system_prompt, user_prompt):
        import requests
        r = requests.post(
            self.base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "temperature": 0.1, "max_tokens": 512,
                  "messages": [{"role": "system", "content": system_prompt},
                               {"role": "user", "content": user_prompt}]},
            timeout=90,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------ 内置模拟器
    def _simulate(self, text):
        feats = self._parse_features(text)
        rules = self._parse_rules(text)
        if feats is None:
            return json.dumps({"defect_type": "none", "severity": "none", "reason": "未解析到感知特征"})

        score = float(feats.get("anomaly_score_pct", 0))
        geo = float(feats.get("geo_distortion", 0))
        area = float(feats.get("max_region_area", 0))
        nreg = int(feats.get("n_suspicious_regions", 0))
        elong = float(feats.get("max_elongation", 0))
        area_pct = abs(float(feats.get("area_diff_signed_pct", 0)))
        part = feats.get("part", "screw")

        # ---- 类型规则树（顺序敏感，代表"质检员的判定习惯"）----
        if nreg >= rules["rust_nreg_th"]:
            dtype = "rust"
        elif part == "screw" and score >= rules["bent_score_th"]:
            dtype = "bent"
        elif geo >= rules["geo_th"] and area_pct >= rules["missing_area_th"]:
            dtype = "missing"
        elif elong >= rules["elong_th"] and area >= rules["scratch_area_min"]:
            dtype = "scratch"
        elif score >= rules["color_score_th"] and area >= rules["color_area_min"] and elong < rules["elong_th"]:
            dtype = "color"
        else:
            dtype = "none"

        if dtype == "none":
            return json.dumps({"defect_type": "none", "severity": "none",
                               "reason": f"score={score:.2f}, nreg={nreg}, elong={elong:.1f}, geo={geo:.2f} "
                                         f"均未达到各类缺陷判据 -> 良品"})

        # ---- 严重度：按类型查 SOP 阈值表 ----
        proxy = {"scratch": area, "color": area, "rust": score,
                 "missing": area_pct, "bent": score}[dtype]
        t1, t2 = rules[f"sev_{dtype}_1"], rules[f"sev_{dtype}_2"]
        severity = "mild" if proxy < t1 else ("moderate" if proxy < t2 else "critical")

        return json.dumps({
            "defect_type": dtype, "severity": severity,
            "reason": f"score={score:.2f}, nreg={nreg}, elong={elong:.1f}, geo={geo:.2f}, "
                      f"area={area:.0f}, |area%|={area_pct:.2f} -> {dtype}/{severity}"
        }, ensure_ascii=False)

    # ------------------------------------------------------------------ 解析
    @staticmethod
    def _parse_features(text):
        m = re.search(r"FEATURES:\s*(\{)", text)
        if not m:
            return None
        s = text[m.start(1):]
        depth = 0
        for i, ch in enumerate(s):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[:i + 1])
                    except json.JSONDecodeError:
                        return None
        return None

    @staticmethod
    def _parse_rules(text):
        m = re.search(r"RULES:(.*?)END RULES", text, re.S)
        block = m.group(1) if m else ""
        rules = dict(DEFAULT_RULES)
        for k, v in re.findall(r"(\w+)\s*=\s*(-?[\d.]+)", block):
            if k in rules:
                rules[k] = float(v)
        return rules

# -*- coding: utf-8 -*-
"""诊断脚本：看感知层在不同缺陷/严重度下的特征分布，用于标定判据阈值"""
import random
import numpy as np
from qc_demo.synthetic_data import generate_one, DEFECTS_BY_PART, SEVERITIES
from qc_demo.perception import PerceptionModule

random.seed(11); np.random.seed(11)
p = PerceptionModule()
hdr = f"{'part':6} {'defect':9} {'sev':9} {'score':>7} {'geo':>7} {'area%':>7} {'nreg':>5} {'maxA':>6} {'elong':>6}"
print(hdr)
for part in ("screw", "nut"):
    for defect in ["none"] + DEFECTS_BY_PART[part]:
        for sev in (["none"] if defect == "none" else SEVERITIES):
            rows = []
            for _ in range(5):
                img, lab = generate_one(part, defect, sev)
                r = p.analyze(img, part)
                rows.append((r["anomaly_score_pct"], r["geo_distortion"], r["area_diff_signed_pct"],
                             r["n_suspicious_regions"], r["max_region_area"], r["max_elongation"]))
            a = np.array(rows, dtype=float).mean(axis=0)
            print(f"{part:6} {defect:9} {sev:9} {a[0]:7.2f} {a[1]:7.2f} {a[2]:7.2f} {a[3]:5.1f} {a[4]:6.0f} {a[5]:6.1f}")

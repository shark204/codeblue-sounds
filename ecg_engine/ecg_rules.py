#!/usr/bin/env python3
"""
ecg_rules.py — Code Blue Clinical ECG Diagnostic Rules Engine
=============================================================
Loads all measurement units, thresholds, lead territories, and clinical
explanations dynamically from `ecg_criteria.json`. Modifying any value in
`ecg_criteria.json` immediately changes engine decisions without editing code.

Features:
- Computes Heart Rate (from RR ms/sec/boxes or 10s strip count)
- Computes QTc via Bazett, Fridericia, and Hodges formulas (sex-specific limits)
- Evaluates electrical axis (Normal, LAD, LAFB, RAD, LPFB, Extreme/NW)
- Detects STEMI (contiguous leads, age/sex V2-V3 cutoffs, reciprocal STD,
  culprit artery, Posterior MI, RV MI, LMCA/aVR, Sgarbossa LBBB/paced)
- Detects Wellens' syndrome & NSTEMI/Ischemia
- Detects Arrhythmias (VF/Asystole, Torsades, VT/WCT via Brugada & Vereckei,
  Pre-excited AF, AF, AF with RVR, Atrial Flutter, SVT/AVNRT/AVRT, MAT, WPW)
- Detects AV Blocks (1st, Mobitz I, Mobitz II, 2:1/High-grade, 3rd/Complete)
- Detects BBB & Fascicular Blocks (RBBB, LBBB, LAFB, LPFB, Bi/Trifascicular)
- Detects Chamber Enlargement (LVH Sokolow-Lyon/Cornell, RVH, LAA, RAA)
- Detects Electrolytes, Channelopathies, Pericarditis/Tamponade, PE, Toxicity,
  Pacemaker rhythms, and RA-LA Lead Reversal
- Sorts CRITICAL and URGENT diagnoses FIRST with bilingual (AR/EN) clinical
  explanations and immediate management protocols.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_CRITERIA_PATH = Path(__file__).resolve().parent / "ecg_criteria.json"


class ECGRulesEngine:
    """Dynamic clinical ECG interpretation engine backed by ecg_criteria.json."""

    def __init__(self, criteria_path: Optional[str | Path] = None) -> None:
        self.criteria_path = Path(criteria_path) if criteria_path else DEFAULT_CRITERIA_PATH
        with open(self.criteria_path, "r", encoding="utf-8") as fh:
            self.criteria: Dict[str, Any] = json.load(fh)
        self.units = self.criteria["units_and_calibration"]
        self.hr_cfg = self.criteria["heart_rate"]
        self.int_cfg = self.criteria["intervals_ms"]
        self.axis_cfg = self.criteria["axis_degrees"]
        self.lead_cfg = self.criteria["lead_groups_and_territories"]
        self.st_cfg = self.criteria["st_and_ischemia_criteria"]
        self.hyp_cfg = self.criteria["hypertrophy_and_voltage_criteria"]
        self.arr_cfg = self.criteria["arrhythmia_and_conduction_criteria"]
        self.urgency_cfg = self.criteria["urgency_levels"]
        self.kb = self.criteria["conditions_knowledge_base"]

    # ------------------------------------------------------------------
    # 1. Core Calculations (Units, HR, QTc, Axis)
    # ------------------------------------------------------------------
    def boxes_to_ms(self, small_boxes: Optional[float] = None, large_boxes: Optional[float] = None) -> Optional[float]:
        if small_boxes is not None:
            return round(small_boxes * self.units["small_box_ms"], 2)
        if large_boxes is not None:
            return round(large_boxes * self.units["large_box_ms"], 2)
        return None

    def compute_heart_rate(self, ecg: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], str]:
        """Return (hr_bpm, rr_ms, calculation_method)."""
        if ecg.get("hr_bpm") is not None:
            hr = float(ecg["hr_bpm"])
            rr_ms = round(60000.0 / hr, 1) if hr > 0 else None
            return round(hr, 1), rr_ms, "direct_hr_bpm"
        if ecg.get("rr_ms") is not None and float(ecg["rr_ms"]) > 0:
            rr_ms = float(ecg["rr_ms"])
            return round(60000.0 / rr_ms, 1), round(rr_ms, 1), "60000 / rr_ms"
        if ecg.get("rr_sec") is not None and float(ecg["rr_sec"]) > 0:
            rr_sec = float(ecg["rr_sec"])
            return round(60.0 / rr_sec, 1), round(rr_sec * 1000.0, 1), "60 / rr_sec"
        if ecg.get("rr_small_boxes") is not None and float(ecg["rr_small_boxes"]) > 0:
            sb = float(ecg["rr_small_boxes"])
            rr_ms = sb * self.units["small_box_ms"]
            return round(60000.0 / rr_ms, 1), round(rr_ms, 1), "1500 / rr_small_boxes"
        if ecg.get("rr_large_boxes") is not None and float(ecg["rr_large_boxes"]) > 0:
            lb = float(ecg["rr_large_boxes"])
            rr_ms = lb * self.units["large_box_ms"]
            return round(60000.0 / rr_ms, 1), round(rr_ms, 1), "300 / rr_large_boxes"
        if ecg.get("qrs_count_10s") is not None:
            hr = float(ecg["qrs_count_10s"]) * (60.0 / self.units["standard_strip_duration_sec"])
            rr_ms = round(60000.0 / hr, 1) if hr > 0 else None
            return round(hr, 1), rr_ms, "qrs_count_10s * 6"
        return None, None, "unavailable"

    def compute_qtc(self, qt_ms: Optional[float], rr_ms: Optional[float], hr_bpm: Optional[float]) -> Dict[str, Optional[float]]:
        """Compute QTc using Bazett, Fridericia, and Hodges formulas."""
        result: Dict[str, Optional[float]] = {
            "qt_ms": qt_ms,
            "bazett_ms": None,
            "fridericia_ms": None,
            "hodges_ms": None,
            "primary_qtc_ms": None,
            "primary_formula": None,
        }
        if qt_ms is None or qt_ms <= 0:
            return result
        if rr_ms is None and hr_bpm and hr_bpm > 0:
            rr_ms = 60000.0 / hr_bpm
        if rr_ms is None or rr_ms <= 0:
            return result

        rr_sec = rr_ms / 1000.0
        qt_sec = qt_ms / 1000.0
        hr_eff = hr_bpm if hr_bpm else (60.0 / rr_sec)

        bazett = round((qt_sec / math.sqrt(rr_sec)) * 1000.0, 1)
        fridericia = round((qt_sec / (rr_sec ** (1.0 / 3.0))) * 1000.0, 1)
        hodges = round(qt_ms + 1.75 * (hr_eff - 60.0), 1)

        qt_cfg = self.int_cfg["qt_qtc"]
        default_formula = qt_cfg.get("default_formula", "bazett")
        result["bazett_ms"] = bazett
        result["fridericia_ms"] = fridericia
        result["hodges_ms"] = hodges

        formula_map = {"bazett": bazett, "fridericia": fridericia, "hodges": hodges}
        result["primary_formula"] = default_formula
        result["primary_qtc_ms"] = formula_map.get(default_formula, bazett)
        return result

    def classify_axis(self, axis_deg: Optional[float]) -> Optional[str]:
        if axis_deg is None:
            return None
        ax = float(axis_deg)
        cfg = self.axis_cfg
        if cfg["normal_min"] <= ax <= cfg["normal_max"]:
            return "Normal Axis"
        if cfg["normal_max"] < ax <= cfg["normal_extended_max"]:
            return "Borderline Right Axis (Normal Extended)"
        if cfg["extreme_nw_min"] <= ax <= cfg["extreme_nw_max"]:
            return "Extreme Axis Deviation (Northwest / No Man's Land)"
        if ax < cfg["normal_min"]:
            return "Left Axis Deviation (LAD)"
        if ax > cfg["normal_extended_max"]:
            return "Right Axis Deviation (RAD)"
        return "Indeterminate Axis"

    # ------------------------------------------------------------------
    # 2. Finding Builder
    # ------------------------------------------------------------------
    def _make_finding(
        self,
        code: str,
        evidence: List[str],
        extra: Optional[Dict[str, Any]] = None,
        urgency_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        kb_entry = self.kb.get(code, {})
        urgency = urgency_override or kb_entry.get("urgency", "ABNORMAL")
        urg_meta = self.urgency_cfg.get(urgency, self.urgency_cfg["ABNORMAL"])
        finding: Dict[str, Any] = {
            "code": code,
            "urgency": urgency,
            "priority": urg_meta["priority"],
            "urgency_label_en": urg_meta["label_en"],
            "urgency_label_ar": urg_meta["label_ar"],
            "name_en": kb_entry.get("name_en", code),
            "name_ar": kb_entry.get("name_ar", code),
            "evidence": evidence,
            "explanation_ar": kb_entry.get("explanation_ar", ""),
            "explanation_en": kb_entry.get("explanation_en", ""),
            "immediate_action_ar": kb_entry.get("immediate_action_ar", ""),
            "immediate_action_en": kb_entry.get("immediate_action_en", ""),
        }
        if extra:
            finding["details"] = extra
        return finding

    # ------------------------------------------------------------------
    # 3. Diagnostic Rule Modules
    # ------------------------------------------------------------------
    def _st_threshold_for_lead(self, lead: str, age: Optional[int], sex: Optional[str]) -> float:
        ste_cfg = self.st_cfg["stemi_j_point_elevation_mm"]
        if lead in ("V7", "V8", "V9"):
            return float(ste_cfg["posterior_v7_v9"])
        if lead in ("V3R", "V4R", "V5R", "V6R"):
            return float(ste_cfg["rv_v4r"])
        if lead in ("V2", "V3"):
            s = (sex or "").strip().lower()
            if s in ("f", "female", "woman"):
                return float(ste_cfg["v2_v3_female_any_age"])
            if s in ("m", "male", "man"):
                if age is not None and int(age) < int(ste_cfg["age_cutoff_male_v2_v3"]):
                    return float(ste_cfg["v2_v3_male_under_40"])
                return float(ste_cfg["v2_v3_male_40_and_over"])
            return float(ste_cfg["v2_v3_unknown_demographics"])
        return float(ste_cfg["default_all_other_leads"])

    def _evaluate_stemi_and_ischemia(
        self,
        ecg: Dict[str, Any],
        has_lbbb: bool,
        is_paced: bool,
        findings: List[Dict[str, Any]],
    ) -> None:
        st_map: Dict[str, float] = ecg.get("st_elevation_mm", {}) or {}
        t_inv_leads: List[str] = ecg.get("t_wave_inversion_leads", []) or []
        age = ecg.get("age")
        sex = ecg.get("sex")

        # --- Sgarbossa for LBBB or RV Paced ---
        if has_lbbb or is_paced:
            sg_cfg = self.st_cfg["sgarbossa_lbbb_or_paced"]
            score = 0
            sg_ev: List[str] = []
            if ecg.get("concordant_ste_mm", 0) >= sg_cfg["concordant_ste_mm"]:
                score += int(sg_cfg["concordant_ste_points"])
                sg_ev.append(f"Concordant STE {ecg['concordant_ste_mm']} mm (+{sg_cfg['concordant_ste_points']} pts)")
            if ecg.get("concordant_std_v1_v3_mm", 0) <= sg_cfg["concordant_std_v1_v3_mm"]:
                score += int(sg_cfg["concordant_std_v1_v3_points"])
                sg_ev.append(f"Concordant STD in V1-V3 {ecg['concordant_std_v1_v3_mm']} mm (+{sg_cfg['concordant_std_v1_v3_points']} pts)")
            if ecg.get("discordant_ste_mm", 0) >= sg_cfg["discordant_ste_mm"]:
                score += int(sg_cfg["discordant_ste_points"])
                sg_ev.append(f"Discordant STE {ecg['discordant_ste_mm']} mm (+{sg_cfg['discordant_ste_points']} pts)")
            smith_ratio = ecg.get("modified_smith_st_s_ratio")
            smith_pos = smith_ratio is not None and abs(float(smith_ratio)) >= float(sg_cfg["modified_smith_st_s_ratio"])
            if smith_pos:
                sg_ev.append(f"Modified Smith ST/S ratio = {smith_ratio} (cutoff >= {sg_cfg['modified_smith_st_s_ratio']})")
            if score >= int(sg_cfg["diagnostic_score_min"]) or smith_pos:
                findings.append(
                    self._make_finding(
                        "SGARBOSSA_POSITIVE",
                        sg_ev,
                        {"sgarbossa_score": score, "modified_smith_positive": smith_pos},
                    )
                )
            return

        # --- Acute Pericarditis Check (before labeling diffuse STE as STEMI) ---
        peric_cfg = self.arr_cfg["pericarditis"]
        concave_ste = bool(ecg.get("st_morphology_concave", False))
        pr_dep = bool(ecg.get("pr_depression", False))
        v6_st_t = ecg.get("v6_st_t_ratio")
        elevated_standard_leads = [
            ld for ld, val in st_map.items()
            if ld in self.lead_cfg["all_standard_leads"] and ld != "aVR" and val >= self._st_threshold_for_lead(ld, age, sex)
        ]
        depressed_leads = [
            ld for ld, val in st_map.items()
            if ld != "aVR" and ld != "V1" and val <= self.st_cfg["nstemi_ischemia_st_depression_mm"]
        ]
        is_pericarditis = (
            len(elevated_standard_leads) >= int(peric_cfg["min_diffuse_ste_leads"])
            and len(depressed_leads) == 0
            and (concave_ste or pr_dep or (v6_st_t is not None and float(v6_st_t) > float(peric_cfg["st_t_ratio_v6_min"])))
        )
        if is_pericarditis:
            ev = [f"Diffuse ST elevation in {len(elevated_standard_leads)} leads ({', '.join(elevated_standard_leads)}) without reciprocal ST depression"]
            if concave_ste:
                ev.append("Concave-upward ST morphology")
            if pr_dep:
                ev.append("PR-segment depression (and/or PR elevation in aVR)")
            if v6_st_t is not None:
                ev.append(f"V6 ST/T ratio = {v6_st_t} (> {peric_cfg['st_t_ratio_v6_min']})")
            findings.append(self._make_finding("ACUTE_PERICARDITIS", ev))
            return

        # --- LMCA / 3-Vessel Critical Ischemia (aVR STE >= 1mm + diffuse STD >= 6 leads) ---
        avr_ste = float(st_map.get("aVR", 0.0))
        std_count_all = [ld for ld, val in st_map.items() if ld != "aVR" and val <= self.st_cfg["nstemi_ischemia_st_depression_mm"]]
        if avr_ste >= float(self.st_cfg["lmca_avr_ste_min_mm"]) and len(std_count_all) >= int(self.st_cfg["lmca_diffuse_std_min_leads"]):
            ev = [
                f"ST elevation in aVR = +{avr_ste} mm (cutoff >= {self.st_cfg['lmca_avr_ste_min_mm']} mm)",
                f"Diffuse ST depression in {len(std_count_all)} leads ({', '.join(std_count_all)})",
            ]
            if avr_ste >= float(st_map.get("V1", 0.0)):
                ev.append("aVR STE >= V1 STE (strongly favors LMCA over proximal LAD)")
            findings.append(self._make_finding("LMCA_OCCLUSION", ev))

        # --- Contiguous Territory STEMI Detection ---
        territories = self.lead_cfg["contiguous_territories"]
        matched_territories: List[Dict[str, Any]] = []
        primary_check_order = ["septal", "anterior", "lateral_low_apical", "lateral_high", "inferior"]

        for t_name in primary_check_order:
            t_info = territories[t_name]
            t_leads = t_info["leads"]
            qualifying = []
            for ld in t_leads:
                val = float(st_map.get(ld, 0.0))
                thresh = self._st_threshold_for_lead(ld, age, sex)
                if val >= thresh:
                    qualifying.append((ld, val, thresh))
            if len(qualifying) >= int(t_info["min_contiguous_leads"]):
                recip = [
                    f"{rl} ({st_map[rl]} mm)"
                    for rl in t_info.get("reciprocal_leads", [])
                    if float(st_map.get(rl, 0.0)) <= self.st_cfg["nstemi_ischemia_st_depression_mm"]
                ]
                culprit = t_info["artery"]
                if t_name == "inferior":
                    ste_iii = float(st_map.get("III", 0.0))
                    ste_ii = float(st_map.get("II", 0.0))
                    if ste_iii > ste_ii:
                        culprit = "RCA (STE III > II)"
                    elif ste_ii >= ste_iii:
                        culprit = "LCx likely (STE II >= III)"
                matched_territories.append({
                    "territory": t_name,
                    "leads": [f"{ld} (+{v}mm >= {th}mm)" for ld, v, th in qualifying],
                    "artery": culprit,
                    "reciprocal": recip,
                })

        # Also check anteroseptal / anterolateral combined if individual pairs had 1+1 adjacent (e.g. V2+V3 or V4+V5)
        if not matched_territories:
            for combined_name in ("anteroseptal", "anterolateral"):
                t_info = territories[combined_name]
                t_leads = t_info["leads"]
                qualifying = []
                for idx, ld in enumerate(t_leads):
                    val = float(st_map.get(ld, 0.0))
                    thresh = self._st_threshold_for_lead(ld, age, sex)
                    if val >= thresh:
                        qualifying.append((idx, ld, val, thresh))
                # Ensure at least two are contiguous in the ordered list
                has_adj = any(qualifying[i + 1][0] - qualifying[i][0] == 1 for i in range(len(qualifying) - 1))
                if len(qualifying) >= 2 and has_adj:
                    recip = [
                        f"{rl} ({st_map[rl]} mm)"
                        for rl in t_info.get("reciprocal_leads", [])
                        if float(st_map.get(rl, 0.0)) <= self.st_cfg["nstemi_ischemia_st_depression_mm"]
                    ]
                    matched_territories.append({
                        "territory": combined_name,
                        "leads": [f"{ld} (+{v}mm >= {th}mm)" for _, ld, v, th in qualifying],
                        "artery": t_info["artery"],
                        "reciprocal": recip,
                    })
                    break

        if matched_territories:
            ev: List[str] = []
            for mt in matched_territories:
                line = f"Territory [{mt['territory'].upper()}]: {', '.join(mt['leads'])} | Suspected Artery: {mt['artery']}"
                if mt["reciprocal"]:
                    line += f" | Reciprocal STD: {', '.join(mt['reciprocal'])}"
                ev.append(line)
            findings.append(self._make_finding("STEMI", ev, {"territories": matched_territories}))

        # --- Posterior STEMI (V7-V9 STE >= 0.5mm OR V1-V3 horizontal STD + tall R) ---
        post_leads = [ld for ld in ("V7", "V8", "V9") if float(st_map.get(ld, 0.0)) >= self._st_threshold_for_lead(ld, age, sex)]
        v1_v3_std = [ld for ld in ("V1", "V2", "V3") if float(st_map.get(ld, 0.0)) <= float(self.st_cfg["posterior_mi_v1_v3_st_depression_mm"])]
        if len(post_leads) >= 2 or (len(v1_v3_std) >= 2 and (ecg.get("tall_r_v1_v2") or ecg.get("upright_t_v1_v3"))):
            ev = []
            if post_leads:
                ev.append(f"Posterior leads STE >= {self.st_cfg['stemi_j_point_elevation_mm']['posterior_v7_v9']} mm in {', '.join(post_leads)}")
            if v1_v3_std:
                ev.append(f"Mirror ST depression in {', '.join(v1_v3_std)}")
            if ecg.get("tall_r_v1_v2"):
                ev.append("Tall R wave in V1-V2 (posterior Q-wave mirror)")
            findings.append(self._make_finding("POSTERIOR_STEMI", ev))

        # --- Right Ventricular MI (V4R >= 1mm) ---
        rv_leads = [ld for ld in ("V3R", "V4R", "V5R", "V6R") if float(st_map.get(ld, 0.0)) >= self._st_threshold_for_lead(ld, age, sex)]
        if rv_leads:
            ev = [f"Right precordial ST elevation in {', '.join(f'{ld} (+{st_map[ld]} mm)' for ld in rv_leads)}"]
            findings.append(self._make_finding("RV_STEMI", ev))

        # --- Wellens' Syndrome ---
        wellens_type = (ecg.get("wellens_pattern") or "").strip().lower()
        t_amp_map = ecg.get("t_wave_amplitude_mm", {}) or {}
        wellens_leads = [
            ld for ld in self.st_cfg["wellens_syndrome"]["target_leads"]
            if float(t_amp_map.get(ld, 0.0)) <= float(self.st_cfg["wellens_syndrome"]["deep_twi_mm"])
        ]
        if wellens_type in ("type_a", "type_b", "biphasic", "deep_symmetric") or (len(wellens_leads) >= 2 and not matched_territories):
            ev = []
            if wellens_type:
                ev.append(f"Wellens morphology: {wellens_type} in V2-V3/V4")
            if wellens_leads:
                ev.append(f"Deep symmetric T-wave inversion (<= {self.st_cfg['wellens_syndrome']['deep_twi_mm']} mm) in {', '.join(wellens_leads)}")
            findings.append(self._make_finding("WELLENS_SYNDROME", ev))

        # --- NSTEMI / Ischemia (if not already STEMI / LMCA / Posterior / Wellens) ---
        already_acs = any(f["code"] in ("STEMI", "LMCA_OCCLUSION", "POSTERIOR_STEMI", "WELLENS_SYNDROME") for f in findings)
        if not already_acs:
            isch_std = [ld for ld, val in st_map.items() if ld != "aVR" and val <= self.st_cfg["nstemi_ischemia_st_depression_mm"]]
            if len(isch_std) >= 2 or len(t_inv_leads) >= 2:
                ev = []
                if len(isch_std) >= 2:
                    ev.append(f"ST depression (<= {self.st_cfg['nstemi_ischemia_st_depression_mm']} mm) in {', '.join(f'{ld} ({st_map[ld]} mm)' for ld in isch_std)}")
                if len(t_inv_leads) >= 2:
                    ev.append(f"Ischemic T-wave inversions in {', '.join(t_inv_leads)}")
                findings.append(self._make_finding("NSTEMI_ISCHEMIA", ev))

    def _evaluate_arrhythmias(
        self,
        ecg: Dict[str, Any],
        hr_bpm: Optional[float],
        qrs_ms: Optional[float],
        pr_ms: Optional[float],
        qtc_ms: Optional[float],
        axis_deg: Optional[float],
        findings: List[Dict[str, Any]],
    ) -> None:
        rhythm_hint = (ecg.get("rhythm_type") or "").strip().lower()
        regular = ecg.get("rhythm_regular", True)
        p_present = ecg.get("p_waves_present", True)
        wide_qrs = qrs_ms is not None and qrs_ms >= float(self.int_cfg["qrs"]["wide_qrs_min_ms"])

        # 1. Cardiac Arrest: VF / Asystole
        if rhythm_hint in ("vf", "ventricular_fibrillation", "asystole"):
            findings.append(
                self._make_finding("VF_OR_ASYSTOLE", [f"Arrest rhythm detected: {rhythm_hint.upper()}"])
            )
            return

        # 2. Torsades de Pointes
        if rhythm_hint in ("torsades", "torsades_de_pointes", "tdp") or (
            ecg.get("polymorphic_vt") and (qtc_ms is not None and qtc_ms >= float(self.int_cfg["qt_qtc"]["normal_upper_female_ms"]))
        ):
            ev = ["Polymorphic VT with twisting QRS axis around isoelectric baseline"]
            if qtc_ms:
                ev.append(f"Baseline/associated QTc = {qtc_ms} ms")
            findings.append(self._make_finding("TORSADES_DE_POINTES", ev))
            return

        # 3. Pre-excited AF (Irregular + Wide QRS + rapid/delta/WPW)
        shortest_rr = ecg.get("shortest_rr_ms")
        if (
            (not regular and wide_qrs and (ecg.get("delta_wave") or ecg.get("varying_qrs_morphology") or (shortest_rr is not None and float(shortest_rr) <= float(self.arr_cfg["af"]["preexcited_af_shortest_rr_ms"]))))
            or rhythm_hint == "preexcited_af"
        ):
            ev = ["Irregularly irregular wide-complex tachycardia with pre-excitation"]
            if shortest_rr is not None:
                ev.append(f"Shortest RR interval = {shortest_rr} ms (high-risk cutoff <= {self.arr_cfg['af']['preexcited_af_shortest_rr_ms']} ms)")
            if hr_bpm:
                ev.append(f"Ventricular rate = {hr_bpm} bpm")
            findings.append(self._make_finding("PREEXCITED_AF", ev))
            return

        # 4. Wide-Complex Tachycardia / VT (Regular, HR >= 100, QRS >= 120 ms)
        wct_cfg = self.arr_cfg["wct_vt_criteria"]
        if (
            (hr_bpm is not None and hr_bpm >= float(wct_cfg["min_rate_bpm"]) and wide_qrs and regular and not ecg.get("known_svt_with_aberrancy"))
            or rhythm_hint in ("vt", "ventricular_tachycardia", "monomorphic_vt")
        ):
            vt_ev = [f"Regular wide-complex tachycardia: HR = {hr_bpm} bpm, QRS = {qrs_ms} ms"]
            if ecg.get("av_dissociation"):
                vt_ev.append("AV dissociation present (~100% specific for VT)")
            if ecg.get("capture_or_fusion_beats"):
                vt_ev.append("Capture and/or fusion beats present (diagnostic of VT)")
            if ecg.get("precordial_concordance"):
                vt_ev.append(f"Precordial concordance ({ecg['precordial_concordance']}) — Brugada Step 1 positive")
            if ecg.get("rs_interval_ms") is not None and float(ecg["rs_interval_ms"]) > float(wct_cfg["brugada_rs_interval_ms"]):
                vt_ev.append(f"RS interval = {ecg['rs_interval_ms']} ms (> {wct_cfg['brugada_rs_interval_ms']} ms, Brugada Step 2 positive)")
            if ecg.get("avr_initial_r_wave") or (ecg.get("avr_vi_vt_ratio") is not None and float(ecg["avr_vi_vt_ratio"]) <= float(wct_cfg["vereckei_vi_vt_ratio_max"])):
                vt_ev.append("Vereckei aVR algorithm positive for VT")
            if axis_deg is not None and self.axis_cfg["extreme_nw_min"] <= float(axis_deg) <= self.axis_cfg["extreme_nw_max"]:
                vt_ev.append(f"Extreme Northwest axis ({axis_deg}°) strongly favors VT")
            findings.append(self._make_finding("VENTRICULAR_TACHYCARDIA", vt_ev))

        # 5. Atrial Flutter (Sawtooth F-waves 240-350 bpm)
        flutter_rate = ecg.get("flutter_wave_rate_bpm")
        if ecg.get("sawtooth_flutter_waves") or rhythm_hint == "atrial_flutter" or (
            flutter_rate is not None and float(self.arr_cfg["atrial_flutter"]["flutter_wave_rate_min_bpm"]) <= float(flutter_rate) <= float(self.arr_cfg["atrial_flutter"]["flutter_wave_rate_max_bpm"])
        ):
            ev = ["Sawtooth atrial flutter waves (F-waves)"]
            if flutter_rate:
                ev.append(f"Atrial flutter rate = {flutter_rate} bpm")
            if hr_bpm:
                ev.append(f"Ventricular rate = {hr_bpm} bpm")
            findings.append(self._make_finding("ATRIAL_FLUTTER", ev))

        # 6. Atrial Fibrillation (Irregularly irregular + absent discrete P waves)
        elif (not regular and not p_present) or ecg.get("fibrillatory_waves") or rhythm_hint in ("af", "afib", "atrial_fibrillation"):
            ev = ["Irregularly irregular RR intervals", "Absent discrete P waves (fibrillatory baseline)"]
            if hr_bpm is not None:
                ev.append(f"Ventricular rate = {hr_bpm} bpm")
            if hr_bpm is not None and hr_bpm > float(self.arr_cfg["af"]["rvr_min_bpm"]):
                findings.append(self._make_finding("ATRIAL_FIBRILLATION_RVR", ev))
            else:
                findings.append(self._make_finding("ATRIAL_FIBRILLATION", ev))

        # 7. Multifocal Atrial Tachycardia (MAT)
        elif (
            ecg.get("distinct_p_morphologies", 1) >= int(self.arr_cfg["mat"]["min_distinct_p_morphologies"])
            and hr_bpm is not None and hr_bpm > float(self.arr_cfg["mat"]["min_rate_bpm"])
        ) or rhythm_hint == "mat":
            ev = [
                f"Irregular tachycardia (HR = {hr_bpm} bpm)",
                f"{ecg.get('distinct_p_morphologies', 3)} distinct P-wave morphologies (>= {self.arr_cfg['mat']['min_distinct_p_morphologies']})",
            ]
            findings.append(self._make_finding("MAT", ev))

        # 8. Regular Narrow-Complex SVT (AVNRT / AVRT / AT)
        elif (
            regular
            and hr_bpm is not None and hr_bpm >= float(self.arr_cfg["svt"]["min_rate_bpm"])
            and (qrs_ms is None or qrs_ms <= float(self.arr_cfg["svt"]["max_qrs_ms"]))
            and (not p_present or ecg.get("retrograde_p_waves") or rhythm_hint in ("svt", "avnrt", "avrt", "psvt"))
        ):
            ev = [f"Regular narrow-complex tachycardia: HR = {hr_bpm} bpm, QRS = {qrs_ms} ms"]
            rp_ms = ecg.get("rp_interval_ms")
            if rp_ms is not None:
                subtype = "AVNRT likely (short RP)" if float(rp_ms) < float(self.arr_cfg["svt"]["avnrt_short_rp_max_ms"]) else "AVRT or Atrial Tachycardia likely (longer RP)"
                ev.append(f"RP interval = {rp_ms} ms -> {subtype}")
            findings.append(self._make_finding("SVT_NARROW", ev))

        # 9. WPW Pattern (in sinus rhythm)
        if ecg.get("delta_wave") or (
            pr_ms is not None and pr_ms <= float(self.int_cfg["pr"]["short_pr_max_ms"]) and wide_qrs
        ):
            ev = []
            if pr_ms is not None:
                ev.append(f"Short PR interval = {pr_ms} ms (< {self.int_cfg['pr']['normal_min_ms']} ms)")
            if qrs_ms is not None:
                ev.append(f"Widened QRS = {qrs_ms} ms")
            if ecg.get("delta_wave"):
                ev.append("Delta wave (slurred QRS upstroke) present")
            findings.append(self._make_finding("WPW_PATTERN", ev))

    def _evaluate_av_blocks(
        self,
        ecg: Dict[str, Any],
        pr_ms: Optional[float],
        hr_bpm: Optional[float],
        findings: List[Dict[str, Any]],
    ) -> bool:
        """Returns True if 1st-degree AV block is present (used for trifascicular check)."""
        avb = (ecg.get("av_block_type") or "").strip().lower()
        first_deg = False

        if avb in ("3rd", "third", "complete", "chb") or (ecg.get("av_dissociation") and (hr_bpm is None or hr_bpm < 60)):
            ev = ["Complete AV dissociation: independent atrial P waves and slower ventricular escape rhythm"]
            if ecg.get("atrial_rate_bpm"):
                ev.append(f"Atrial rate = {ecg['atrial_rate_bpm']} bpm > Ventricular escape rate = {hr_bpm} bpm")
            findings.append(self._make_finding("COMPLETE_HEART_BLOCK", ev))
        elif avb in ("mobitz_2", "mobitz_ii", "type_2"):
            ev = ["Constant PR intervals with abrupt dropped QRS complexes (infranodal risk)"]
            if pr_ms:
                ev.append(f"Constant conducted PR = {pr_ms} ms")
            findings.append(self._make_finding("MOBITZ_II_AV_BLOCK", ev))
        elif avb in ("2:1", "2_to_1", "high_grade"):
            ev = [f"AV block pattern: {avb} (multiple or alternating non-conducted P waves)"]
            findings.append(self._make_finding("HIGH_GRADE_OR_2_TO_1_AV_BLOCK", ev))
        elif avb in ("mobitz_1", "mobitz_i", "wenckebach", "type_1") or ecg.get("progressive_pr_prolongation"):
            ev = ["Progressive PR prolongation followed by a dropped QRS complex (Wenckebach periodicity)"]
            findings.append(self._make_finding("MOBITZ_I_AV_BLOCK", ev))
        elif pr_ms is not None and pr_ms >= float(self.int_cfg["pr"]["first_degree_avb_min_ms"]):
            first_deg = True
            ev = [f"PR interval = {pr_ms} ms (normal <= {self.int_cfg['pr']['normal_max_ms']} ms) with 1:1 AV conduction"]
            findings.append(self._make_finding("FIRST_DEGREE_AV_BLOCK", ev))

        return first_deg

    def _evaluate_bbb_and_fascicular(
        self,
        ecg: Dict[str, Any],
        qrs_ms: Optional[float],
        axis_deg: Optional[float],
        has_first_deg_avb: bool,
        findings: List[Dict[str, Any]],
    ) -> Tuple[bool, bool]:
        """Returns (has_lbbb, has_rbbb)."""
        wide = qrs_ms is not None and qrs_ms >= float(self.int_cfg["qrs"]["wide_qrs_min_ms"])
        morph = (ecg.get("qrs_morphology") or "").strip().lower()
        has_rbbb = False
        has_lbbb = False

        if wide and (morph == "rbbb" or (ecg.get("rsr_prime_v1") and ecg.get("wide_s_i_v6"))):
            has_rbbb = True
            ev = [f"QRS = {qrs_ms} ms (>= {self.int_cfg['qrs']['wide_qrs_min_ms']} ms)", "rSR' pattern in V1-V2 and wide slurred S in I/V6"]
            findings.append(self._make_finding("RBBB", ev))
        elif wide and (morph == "lbbb" or (ecg.get("broad_monophasic_r_lateral") and ecg.get("deep_qs_v1"))):
            has_lbbb = True
            ev = [f"QRS = {qrs_ms} ms (>= {self.int_cfg['qrs']['wide_qrs_min_ms']} ms)", "Broad notched R in I/aVL/V5-V6 with absent septal Q and deep QS/rS in V1"]
            findings.append(self._make_finding("LBBB", ev))

        # Fascicular / Bifascicular / Trifascicular
        has_lafb = bool(ecg.get("lafb")) or (axis_deg is not None and self.axis_cfg["lafb_min"] <= float(axis_deg) <= self.axis_cfg["lafb_max"])
        has_lpfb = bool(ecg.get("lpfb")) or (axis_deg is not None and float(axis_deg) >= self.axis_cfg["lpfb_min"] and ecg.get("qr_inferior_rs_lateral"))

        if has_rbbb and (has_lafb or has_lpfb):
            fasc = "LAFB" if has_lafb else "LPFB"
            if has_first_deg_avb:
                findings.append(
                    self._make_finding("TRIFASCICULAR_BLOCK", [f"RBBB + {fasc} (axis {axis_deg}°) + 1st-degree AV block"])
                )
            else:
                findings.append(
                    self._make_finding("BIFASCICULAR_BLOCK", [f"RBBB + {fasc} (axis {axis_deg}°)"])
                )

        return has_lbbb, has_rbbb

    def _evaluate_qtc(self, ecg: Dict[str, Any], qtc_info: Dict[str, Optional[float]], findings: List[Dict[str, Any]]) -> None:
        qtc_ms = qtc_info.get("primary_qtc_ms")
        if qtc_ms is None:
            return
        qt_cfg = self.int_cfg["qt_qtc"]
        sex = (ecg.get("sex") or "").strip().lower()
        upper = float(qt_cfg["normal_upper_female_ms"]) if sex in ("f", "female", "woman") else float(qt_cfg["normal_upper_male_ms"])

        if qtc_ms >= float(qt_cfg["critical_torsades_risk_ms"]):
            findings.append(
                self._make_finding(
                    "LONG_QT_CRITICAL",
                    [f"QTc ({qtc_info['primary_formula']}) = {qtc_ms} ms (>= critical cutoff {qt_cfg['critical_torsades_risk_ms']} ms)"],
                )
            )
        elif qtc_ms > upper:
            findings.append(
                self._make_finding(
                    "LONG_QT",
                    [f"QTc ({qtc_info['primary_formula']}) = {qtc_ms} ms (normal upper limit {upper} ms)"],
                )
            )
        elif qtc_ms <= float(qt_cfg["short_qt_max_ms"]):
            findings.append(
                self._make_finding(
                    "SHORT_QT",
                    [f"QTc ({qtc_info['primary_formula']}) = {qtc_ms} ms (<= short QT cutoff {qt_cfg['short_qt_max_ms']} ms)"],
                )
            )

    def _evaluate_special_conditions(
        self,
        ecg: Dict[str, Any],
        hr_bpm: Optional[float],
        qrs_ms: Optional[float],
        axis_deg: Optional[float],
        findings: List[Dict[str, Any]],
    ) -> None:
        # Brugada
        brug = (ecg.get("brugada_pattern") or "").strip().lower()
        st_map = ecg.get("st_elevation_mm", {}) or {}
        brug_cfg = self.arr_cfg["brugada"]
        v1_v2_ste = max(float(st_map.get("V1", 0.0)), float(st_map.get("V2", 0.0)))
        if brug == "type_1" or (ecg.get("coved_ste_v1_v2") and v1_v2_ste >= float(brug_cfg["type1_coved_ste_min_mm"])):
            findings.append(
                self._make_finding("BRUGADA_TYPE_1", [f"Coved ST elevation >= {brug_cfg['type1_coved_ste_min_mm']} mm (measured {v1_v2_ste} mm) with negative T wave in V1-V2"])
            )
        elif brug == "type_2" or (ecg.get("saddleback_ste_v1_v2") and v1_v2_ste >= float(brug_cfg["type2_saddleback_ste_min_mm"])):
            findings.append(
                self._make_finding("BRUGADA_TYPE_2", [f"Saddleback ST elevation >= {brug_cfg['type2_saddleback_ste_min_mm']} mm (measured {v1_v2_ste} mm) in V1-V2"])
            )

        # Hyperkalemia
        hk_cfg = self.arr_cfg["hyperkalemia"]
        max_t = float(ecg.get("max_t_wave_amplitude_mm", 0.0) or 0.0)
        peaked_t = bool(ecg.get("peaked_t_waves")) or max_t >= float(hk_cfg["peaked_t_wave_amplitude_precordial_mm"])
        if ecg.get("sine_wave_pattern") or (peaked_t and (qrs_ms is not None and qrs_ms >= float(hk_cfg["severe_qrs_widening_ms"]) or ecg.get("flattened_or_absent_p_hyperk"))):
            ev = ["Severe hyperkalemia ECG pattern"]
            if peaked_t:
                ev.append("Tall symmetrically peaked T waves")
            if qrs_ms and qrs_ms >= float(hk_cfg["severe_qrs_widening_ms"]):
                ev.append(f"Widened QRS = {qrs_ms} ms")
            if ecg.get("sine_wave_pattern"):
                ev.append("Sine-wave QRS-T fusion pattern")
            findings.append(self._make_finding("HYPERKALEMIA_SEVERE", ev))
        elif peaked_t:
            findings.append(self._make_finding("HYPERKALEMIA", [f"Symmetrically peaked tented T waves (max T = {max_t} mm)"]))

        # Hypokalemia
        u_mm = float(ecg.get("u_wave_amplitude_mm", 0.0) or 0.0)
        if ecg.get("prominent_u_waves") or u_mm >= float(self.arr_cfg["hypokalemia"]["prominent_u_wave_mm"]):
            findings.append(self._make_finding("HYPOKALEMIA", [f"Prominent U waves (amplitude {u_mm} mm >= {self.arr_cfg['hypokalemia']['prominent_u_wave_mm']} mm) with flattened T/ST depression"]))

        # Hypothermia / Osborn
        if ecg.get("osborn_j_waves"):
            findings.append(self._make_finding("HYPOTHERMIA_OSBORN", ["Distinctive Osborn (J) waves at J-point"]))

        # Cardiac Tamponade (Low voltage + tachycardia + electrical alternans)
        low_v_cfg = self.hyp_cfg["low_voltage"]
        low_voltage = bool(ecg.get("low_voltage")) or (
            ecg.get("max_limb_qrs_mm") is not None and float(ecg["max_limb_qrs_mm"]) < float(low_v_cfg["limb_leads_max_qrs_mm"])
        )
        if low_voltage and (ecg.get("electrical_alternans") or (hr_bpm is not None and hr_bpm > float(self.hr_cfg["normal_max_bpm"]) and ecg.get("pericardial_effusion_suspected"))):
            ev = ["Low QRS voltage"]
            if ecg.get("electrical_alternans"):
                ev.append("Beat-to-beat electrical alternans")
            if hr_bpm:
                ev.append(f"Sinus tachycardia (HR = {hr_bpm} bpm)")
            findings.append(self._make_finding("CARDIAC_TAMPONADE_SUSPECTED", ev))

        # Pulmonary Embolism
        pe_signs = []
        if ecg.get("s1q3t3"):
            pe_signs.append("S1Q3T3 pattern (deep S in I, Q and inverted T in III)")
        if ecg.get("rv_strain_twi_v1_v4"):
            pe_signs.append("T-wave inversions in V1-V4 (acute RV strain)")
        if pe_signs and (hr_bpm is None or hr_bpm >= 95 or ecg.get("s1q3t3")):
            if hr_bpm and hr_bpm > 100:
                pe_signs.insert(0, f"Sinus tachycardia (HR = {hr_bpm} bpm)")
            findings.append(self._make_finding("PULMONARY_EMBOLISM", pe_signs))

        # TCA / Sodium-channel blocker toxicity
        tca_cfg = self.arr_cfg["tca_toxicity"]
        avr_r = float(ecg.get("avr_terminal_r_mm", 0.0) or 0.0)
        avr_rs = float(ecg.get("avr_r_to_s_ratio", 0.0) or 0.0)
        if qrs_ms is not None and qrs_ms >= float(tca_cfg["qrs_min_ms"]) and (avr_r >= float(tca_cfg["avr_terminal_r_min_mm"]) or avr_rs > float(tca_cfg["avr_r_to_s_ratio_min"])):
            findings.append(
                self._make_finding(
                    "TCA_TOXICITY",
                    [f"QRS = {qrs_ms} ms (>= {tca_cfg['qrs_min_ms']} ms)", f"Terminal R in aVR = {avr_r} mm (R/S = {avr_rs})"],
                )
            )

        # LVH (Sokolow-Lyon & Cornell)
        lvh_cfg = self.hyp_cfg["lvh"]
        s_v1 = float(ecg.get("s_v1_mm", 0.0) or 0.0)
        r_v5 = float(ecg.get("r_v5_mm", 0.0) or 0.0)
        r_v6 = float(ecg.get("r_v6_mm", 0.0) or 0.0)
        r_avl = float(ecg.get("r_avl_mm", 0.0) or 0.0)
        s_v3 = float(ecg.get("s_v3_mm", 0.0) or 0.0)
        sex = (ecg.get("sex") or "").strip().lower()
        cornell_thresh = float(lvh_cfg["cornell_female_ravl_plus_sv3_mm"]) if sex in ("f", "female", "woman") else float(lvh_cfg["cornell_male_ravl_plus_sv3_mm"])
        sokolow = s_v1 + max(r_v5, r_v6)
        cornell = r_avl + s_v3
        lvh_ev = []
        if sokolow >= float(lvh_cfg["sokolow_lyon_sv1_plus_rv5_or_rv6_mm"]):
            lvh_ev.append(f"Sokolow-Lyon: S(V1) + max(R V5/V6) = {sokolow} mm (>= {lvh_cfg['sokolow_lyon_sv1_plus_rv5_or_rv6_mm']} mm)")
        if cornell > cornell_thresh:
            lvh_ev.append(f"Cornell: R(aVL) + S(V3) = {cornell} mm (> {cornell_thresh} mm)")
        if r_avl >= float(lvh_cfg["r_avl_min_mm"]):
            lvh_ev.append(f"R(aVL) = {r_avl} mm (>= {lvh_cfg['r_avl_min_mm']} mm)")
        if lvh_ev:
            findings.append(self._make_finding("LVH", lvh_ev))

        # RVH
        rvh_cfg = self.hyp_cfg["rvh"]
        r_v1 = float(ecg.get("r_v1_mm", 0.0) or 0.0)
        r_s_v1 = float(ecg.get("v1_r_to_s_ratio", 0.0) or 0.0)
        if (r_s_v1 > float(rvh_cfg["v1_r_to_s_ratio_min"]) or r_v1 >= float(rvh_cfg["v1_r_amplitude_min_mm"])) and (axis_deg is None or float(axis_deg) >= float(rvh_cfg["rad_min_degrees"])):
            findings.append(
                self._make_finding("RVH", [f"Dominant R in V1 (R = {r_v1} mm, R/S = {r_s_v1})", f"Axis = {axis_deg}°"])
            )

        # Pacemaker
        if ecg.get("pacing_spikes"):
            findings.append(self._make_finding("PACEMAKER_RHYTHM", ["Pacing spikes detected"]))

        # Lead Reversal (RA-LA)
        if ecg.get("negative_p_qrs_lead_i") and ecg.get("normal_precordial_r_progression", True):
            findings.append(
                self._make_finding("LEAD_REVERSAL_RA_LA", ["Global negativity (P, QRS, T) in Lead I with normal precordial R progression (V1->V6)"])
            )

    # ------------------------------------------------------------------
    # 4. Main Interpretation Pipeline
    # ------------------------------------------------------------------
    def interpret(self, ecg: Dict[str, Any]) -> Dict[str, Any]:
        """Run full clinical ECG rule pipeline and return urgent-first structured report."""
        hr_bpm, rr_ms, hr_method = self.compute_heart_rate(ecg)
        pr_ms = float(ecg["pr_ms"]) if ecg.get("pr_ms") is not None else self.boxes_to_ms(ecg.get("pr_small_boxes"))
        qrs_ms = float(ecg["qrs_ms"]) if ecg.get("qrs_ms") is not None else self.boxes_to_ms(ecg.get("qrs_small_boxes"))
        qt_ms = float(ecg["qt_ms"]) if ecg.get("qt_ms") is not None else self.boxes_to_ms(ecg.get("qt_small_boxes"))
        axis_deg = float(ecg["axis_deg"]) if ecg.get("axis_deg") is not None else None

        qtc_info = self.compute_qtc(qt_ms, rr_ms, hr_bpm)
        axis_label = self.classify_axis(axis_deg)
        is_paced = bool(ecg.get("pacing_spikes", False))

        findings: List[Dict[str, Any]] = []

        # Run rule modules
        has_first_deg = self._evaluate_av_blocks(ecg, pr_ms, hr_bpm, findings)
        has_lbbb, has_rbbb = self._evaluate_bbb_and_fascicular(ecg, qrs_ms, axis_deg, has_first_deg, findings)
        self._evaluate_arrhythmias(ecg, hr_bpm, qrs_ms, pr_ms, qtc_info.get("primary_qtc_ms"), axis_deg, findings)
        self._evaluate_stemi_and_ischemia(ecg, has_lbbb, is_paced, findings)
        self._evaluate_qtc(ecg, qtc_info, findings)
        self._evaluate_special_conditions(ecg, hr_bpm, qrs_ms, axis_deg, findings)

        # If no abnormal findings -> Normal Sinus Rhythm (verify normal HR)
        if not findings:
            nsr_ev = []
            if hr_bpm is not None:
                nsr_ev.append(f"HR = {hr_bpm} bpm (normal {self.hr_cfg['normal_min_bpm']}-{self.hr_cfg['normal_max_bpm']})")
            if pr_ms is not None:
                nsr_ev.append(f"PR = {pr_ms} ms")
            if qrs_ms is not None:
                nsr_ev.append(f"QRS = {qrs_ms} ms")
            if qtc_info["primary_qtc_ms"] is not None:
                nsr_ev.append(f"QTc ({qtc_info['primary_formula']}) = {qtc_info['primary_qtc_ms']} ms")
            if axis_label:
                nsr_ev.append(f"Axis = {axis_deg}° ({axis_label})")
            findings.append(self._make_finding("NORMAL_SINUS_RHYTHM", nsr_ev))

        # Deduplicate by code and sort strictly by urgency priority (CRITICAL=1 -> URGENT=2 -> ABNORMAL=3 -> NORMAL=4)
        dedup: Dict[str, Dict[str, Any]] = {}
        for f in findings:
            if f["code"] not in dedup:
                dedup[f["code"]] = f
        sorted_findings = sorted(dedup.values(), key=lambda x: x["priority"])

        urgent_alerts = [f for f in sorted_findings if f["urgency"] in ("CRITICAL", "URGENT")]
        overall_urgency = sorted_findings[0]["urgency"]

        return {
            "overall_urgency": overall_urgency,
            "has_urgent_findings": len(urgent_alerts) > 0,
            "urgent_count": len(urgent_alerts),
            "measurements": {
                "hr_bpm": hr_bpm,
                "hr_method": hr_method,
                "rr_ms": rr_ms,
                "pr_ms": pr_ms,
                "qrs_ms": qrs_ms,
                "qt_ms": qt_ms,
                "qtc_bazett_ms": qtc_info["bazett_ms"],
                "qtc_fridericia_ms": qtc_info["fridericia_ms"],
                "qtc_hodges_ms": qtc_info["hodges_ms"],
                "axis_deg": axis_deg,
                "axis_classification": axis_label,
            },
            "urgent_findings": urgent_alerts,
            "all_findings": sorted_findings,
        }

    def format_clinical_report(self, result: Dict[str, Any], lang: str = "ar") -> str:
        """Render human-readable clinical report with urgent findings at the top."""
        lines: List[str] = []
        m = result["measurements"]
        if lang == "ar":
            lines.append("================ تقرير قراءة رسم القلب (ECG) ================")
            if result["has_urgent_findings"]:
                lines.append(f"⚠️ تنبيه: يوجد {result['urgent_count']} حالة حرجة/عاجلة تستدعي الانتباه الفوري!")
            lines.append(
                f"القياسات: HR={m['hr_bpm']} bpm | PR={m['pr_ms']} ms | QRS={m['qrs_ms']} ms | "
                f"QTc(Bazett)={m['qtc_bazett_ms']} ms | Axis={m['axis_deg']}° ({m['axis_classification']})"
            )
            lines.append("-" * 60)
            for idx, f in enumerate(result["all_findings"], 1):
                lines.append(f"{idx}. [{f['urgency']} — {f['urgency_label_ar']}] {f['name_ar']} ({f['name_en']})")
                lines.append(f"   • الأدلة من الرسم: {' ; '.join(f['evidence'])}")
                lines.append(f"   • الشرح الإكلينيكي: {f['explanation_ar']}")
                lines.append(f"   • التصرف الطبي العاجل: {f['immediate_action_ar']}")
        else:
            lines.append("================ ECG INTERPRETATION REPORT ================")
            if result["has_urgent_findings"]:
                lines.append(f"ALERT: {result['urgent_count']} CRITICAL/URGENT finding(s) detected!")
            lines.append(
                f"Measurements: HR={m['hr_bpm']} bpm | PR={m['pr_ms']} ms | QRS={m['qrs_ms']} ms | "
                f"QTc(Bazett)={m['qtc_bazett_ms']} ms | Axis={m['axis_deg']}° ({m['axis_classification']})"
            )
            lines.append("-" * 60)
            for idx, f in enumerate(result["all_findings"], 1):
                lines.append(f"{idx}. [{f['urgency']}] {f['name_en']}")
                lines.append(f"   • Evidence: {' ; '.join(f['evidence'])}")
                lines.append(f"   • Clinical Explanation: {f['explanation_en']}")
                lines.append(f"   • Immediate Management: {f['immediate_action_en']}")
        return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Code Blue ECG Diagnostic Rules Engine")
    parser.add_argument("--input", "-i", required=True, help="Path to JSON file containing ECG measurements")
    parser.add_argument("--criteria", "-c", default=str(DEFAULT_CRITERIA_PATH), help="Path to ecg_criteria.json")
    parser.add_argument("--lang", "-l", choices=["ar", "en", "json"], default="ar", help="Output format")
    args = parser.parse_args()

    engine = ECGRulesEngine(args.criteria)
    with open(args.input, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    result = engine.interpret(payload)
    if args.lang == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(engine.format_clinical_report(result, lang=args.lang))


if __name__ == "__main__":
    main()

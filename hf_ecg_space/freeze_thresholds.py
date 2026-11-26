#!/usr/bin/env python3
"""
freeze_thresholds.py — اختيار عتبات لكل فئة على val ثم تجميدها وتطبيقها مرة واحدة على test.

ملف الاحتمالات (CSV): filename,truth,p_<CLASS1>,p_<CLASS2>,...   (truth = فئات مفصولة بـ |، وفارغ أو NORM أو NORMAL_SINUS_RHYTHM = سليم)

1) اختيار العتبات على Fold 9 (بعد المعايرة النهائية!):
   python freeze_thresholds.py select --pred fold9_probs.csv --critical ANTERIOR_SEPTAL_MI,INFERIOR_POSTERIOR_MI,LATERAL_MI,HIGH_GRADE_AV_BLOCK --target-sens 0.90 \
          --max-fpr 0.15 --model-hash <sha256 للأوزان> --out frozen.json
2) تطبيق واحد على الاختبار المقفول (يرفض التشغيل الثاني):
   python freeze_thresholds.py apply --pred test_probs.csv --frozen frozen.json
"""
import argparse, csv, hashlib, json, math, os, sys, time

NORMAL_TOKENS = {"NORM", "NORMAL_SINUS_RHYTHM"}


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


def fmt(k, n):
    p, lo, hi = wilson(k, n)
    return "n/a" if n == 0 else f"{100*p:5.1f}% [{100*lo:4.1f}-{100*hi:4.1f}] ({k}/{n})"


def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        classes = [c[2:] for c in rd.fieldnames if c.startswith("p_") and c[2:] not in NORMAL_TOKENS]
        for r in rd:
            truth = {t.strip() for t in (r.get("truth") or "").split("|") if t.strip() and t.strip() not in NORMAL_TOKENS}
            rows.append(dict(file=r["filename"], truth=truth, p={c: float(r["p_" + c]) for c in classes}))
    return rows, classes


def stats_at(rows, c, t):
    pos = [r for r in rows if c in r["truth"]]
    neg = [r for r in rows if c not in r["truth"]]
    tp = sum(1 for r in pos if r["p"][c] >= t)
    fp = sum(1 for r in neg if r["p"][c] >= t)
    return tp, len(pos), fp, len(neg)


def burden(rows, thr, critical):
    normals = [r for r in rows if not r["truth"]]
    if not normals:
        return None
    alerts = [sum(1 for c in critical if c in thr and r["p"].get(c, 0) >= thr[c]) for r in normals]
    return dict(n_normals=len(normals), pct_normals_with_any_critical_alert=100 * sum(1 for a in alerts if a) / len(normals),
                mean_critical_alerts_per_100_normals=100 * sum(alerts) / len(normals))


def cmd_select(a):
    rows, classes = load(a.pred)
    critical = [c for c in a.critical.split(",") if c]
    thr, report = {}, {}
    for c in classes:
        cands = sorted({r["p"][c] for r in rows}, reverse=True)
        best, note = None, ""
        if c in critical:
            for t in cands:                       # أعلى عتبة تحقق الحساسية المطلوبة
                tp, np_, fp, nn = stats_at(rows, c, t)
                if np_ and tp / np_ >= a.target_sens:
                    best = t
                    break
            if best is None:
                note = "INFEASIBLE: no threshold reaches the target sensitivity"
                best = cands[-1] if cands else 0.5
        else:                                     # Youden J
            bj = -1
            for t in cands:
                tp, np_, fp, nn = stats_at(rows, c, t)
                if np_ and nn:
                    j = tp / np_ - fp / nn
                    if j > bj:
                        bj, best = j, t
            best = 0.5 if best is None else best
        tp, np_, fp, nn = stats_at(rows, c, best)
        fpr = fp / nn if nn else float("nan")
        if c in critical and note == "" and nn and fpr > a.max_fpr:
            note = f"TRADE-OFF: sensitivity target met but FPR {100*fpr:.1f}% > max {100*a.max_fpr:.0f}%"
        if np_ < 30:
            note = (note + "; " if note else "") + f"n_pos={np_} < 30 (sensitivity CI is wide)"
        thr[c] = best
        report[c] = dict(threshold=best, critical=c in critical, sens=fmt(tp, np_), fpr=fmt(fp, nn),
                         ppv=fmt(tp, tp + fp), note=note)
    b = burden(rows, thr, critical)
    frozen = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), model_hash=a.model_hash, source_file=os.path.basename(a.pred),
                  source_sha256=hashlib.sha256(open(a.pred, "rb").read()).hexdigest(), target_sens=a.target_sens,
                  max_fpr=a.max_fpr, critical=critical, thresholds=thr, val_report=report, val_alert_burden=b)
    json.dump(frozen, open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nSelected on VAL (n={len(rows)}). Frozen -> {a.out}\n")
    for c, d in report.items():
        print(f"{c:28s} thr={d['threshold']:.3f} {'[CRITICAL]' if d['critical'] else '          '} sens {d['sens']}\n{'':28s} FPR {d['fpr']}  PPV {d['ppv']}" + (f"\n{'':28s} !! {d['note']}" if d['note'] else ""))
    if b:
        print(f"\nAlert burden on {b['n_normals']} NORMAL val ECGs: {b['pct_normals_with_any_critical_alert']:.1f}% get >=1 critical alert; "
              f"{b['mean_critical_alerts_per_100_normals']:.1f} critical alerts per 100 normals")
    print("\nلا تغيّر الأوزان أو المعايرة بعد التجميد، وإلا أعد الاختيار.")


def cmd_apply(a):
    lock = a.frozen + ".test_used.lock"
    if os.path.exists(lock) and not a.force:
        sys.exit(f"REFUSED: the frozen thresholds were already applied to a test set ({lock}). "
                 "A second look at the same test set is no longer a clean test.")
    fz = json.load(open(a.frozen, encoding="utf-8"))
    rows, classes = load(a.pred)
    thr, critical = fz["thresholds"], fz["critical"]
    print(f"\nONE-SHOT TEST (n={len(rows)}) with thresholds frozen on {fz['created']} (model {fz['model_hash']})\n")
    for c in thr:
        if c not in classes:
            print(f"{c:28s} missing from test file"); continue
        tp, np_, fp, nn = stats_at(rows, c, thr[c])
        print(f"{c:28s} thr={thr[c]:.3f} {'[CRITICAL]' if c in critical else '          '} sens {fmt(tp, np_)}\n{'':28s} spec {fmt(nn - fp, nn)}  PPV {fmt(tp, tp + fp)}" + ("  (n_pos<30!)" if np_ < 30 else ""))
    b = burden(rows, thr, critical)
    if b:
        print(f"\nAlert burden on {b['n_normals']} NORMAL test ECGs: {b['pct_normals_with_any_critical_alert']:.1f}% with >=1 critical alert; "
              f"{b['mean_critical_alerts_per_100_normals']:.1f} critical alerts per 100 normals")
    crit_cases = [r for r in rows if r["truth"] & set(critical)]
    if crit_cases:
        missed = sum(1 for r in crit_cases if not any(c in thr and c in r["truth"] and r["p"].get(c, 0) >= thr[c] for c in critical))
        print(f"Critical cases with NO correct critical alert: {fmt(missed, len(crit_cases))}")
    open(lock, "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"\nLocked: {lock}. هذه النتيجة هي النتيجة المعلنة؛ لا تعدّل شيئاً وتعيد التشغيل على نفس الاختبار.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("select")
    s.add_argument("--pred", required=True); s.add_argument("--critical", default="")
    s.add_argument("--target-sens", type=float, default=0.90); s.add_argument("--max-fpr", type=float, default=0.15)
    s.add_argument("--model-hash", default="UNSPECIFIED"); s.add_argument("--out", default="frozen.json")
    p = sp.add_parser("apply")
    p.add_argument("--pred", required=True); p.add_argument("--frozen", required=True); p.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cmd_select(a) if a.cmd == "select" else cmd_apply(a)

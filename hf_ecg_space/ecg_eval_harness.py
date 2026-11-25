#!/usr/bin/env python3
"""
ecg_eval_harness.py — تقييم أي نسخة من نظام التشخيص بأرقام لها فواصل ثقة (بدون مكتبات خارجية).

يحسب لكل نطاق (domain) على حدة:
- Top-1 / Top-3 (الرفض يُحسب خطأ) + دقة الحالات المُجابة فقط، مع Wilson 95% CI
- منحنى التغطية مقابل الدقة (selective accuracy) واختيار الحد على split=val وتطبيقه على split=test
- حساسية/نوعية/PPV لكل فئة، وتقرير الفئات الخطرة (فائتة بثقة / مرفوضة)
- المعايرة: ECE بفاصل ثقة + جدول موثوقية
- مقارنة نسختين على نفس الصور (McNemar الدقيق + فرق الدقة بفاصل ثقة)
- تحذير حجم العينة (كم صورة تحتاج لإثبات الهدف)
"""
import argparse
import csv
import json
import math
import random
import sys


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def fmt(k, n):
    p, lo, hi = wilson(k, n)
    return "n/a (n=0)" if n == 0 else f"{100*p:5.1f}% [{100*lo:4.1f}-{100*hi:4.1f}] ({k}/{n})"


def mcnemar_exact(b, c):
    """two-sided exact p-value; b = only A correct, c = only B correct."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * p)


def n_needed(p_true, lower_target, z=1.96):
    for n in range(20, 20001, 10):
        k = round(p_true * n)
        if wilson(k, n, z)[1] >= lower_target:
            return n
    return None


def load(path):
    rows = []
    if path.lower().endswith(".json"):
        for r in json.load(open(path, encoding="utf-8")):
            preds = [(p["label"], float(p.get("confidence", 0))) for p in r.get("predictions", [])]
            truth = r["truth"] if isinstance(r["truth"], list) else str(r["truth"]).split("|")
            rows.append(
                dict(
                    file=r["filename"],
                    domain=r.get("domain", "all"),
                    split=r.get("split", ""),
                    truth=set(t.strip() for t in truth if t.strip()),
                    preds=preds,
                    abstain=bool(r.get("abstain", False)),
                )
            )
        return rows
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            preds = []
            for i in (1, 2, 3):
                lab, conf = (r.get(f"pred{i}") or "").strip(), r.get(f"conf{i}")
                if lab:
                    preds.append((lab, float(conf) if conf not in (None, "") else 0.0))
            ab = str(r.get("abstain", "0")).strip().lower() in ("1", "true", "yes")
            rows.append(
                dict(
                    file=r["filename"],
                    domain=(r.get("domain") or "all").strip(),
                    split=(r.get("split") or "").strip(),
                    truth=set(t.strip() for t in (r.get("truth") or "").split("|") if t.strip()),
                    preds=preds,
                    abstain=ab,
                )
            )
    return rows


def answered(r):
    return (not r["abstain"]) and len(r["preds"]) > 0


def top1_ok(r):
    return answered(r) and r["preds"][0][0] in r["truth"]


def top3_ok(r):
    return answered(r) and any(l in r["truth"] for l, _ in r["preds"][:3])


def selective_table(rows, target):
    ans = sorted([r for r in rows if answered(r)], key=lambda r: -r["preds"][0][1])
    n = len(rows)
    out, correct = [], 0
    marks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    mi = 0
    for i, r in enumerate(ans, 1):
        correct += 1 if top1_ok(r) else 0
        while mi < len(marks) and i >= marks[mi] * n:
            p, lo, hi = wilson(correct, i)
            out.append((marks[mi], i, correct, p, lo, hi, r["preds"][0][1]))
            mi += 1
    best = None
    for cov, i, k, p, lo, hi, thr in out:
        if p >= target:
            best = (cov, i, k, p, lo, hi, thr)
    return out, best


def pick_threshold(val_rows, target):
    ans = sorted([r for r in val_rows if answered(r)], key=lambda r: -r["preds"][0][1])
    best_thr, correct = None, 0
    for i, r in enumerate(ans, 1):
        correct += 1 if top1_ok(r) else 0
        if correct / i >= target:
            best_thr = r["preds"][0][1]
    return best_thr


def ece(rows, bins=10):
    ans = [r for r in rows if answered(r)]
    if not ans:
        return float("nan"), []
    tab, e = [], 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [r for r in ans if (lo <= r["preds"][0][1] < hi) or (b == bins - 1 and r["preds"][0][1] == 1.0)]
        if not sel:
            continue
        acc = sum(top1_ok(r) for r in sel) / len(sel)
        conf = sum(r["preds"][0][1] for r in sel) / len(sel)
        e += len(sel) / len(ans) * abs(acc - conf)
        tab.append((lo, hi, len(sel), conf, acc))
    return e, tab


def boot_ece(rows, n_boot=1000, seed=1):
    ans = [r for r in rows if answered(r)]
    if len(ans) < 5:
        return (float("nan"), float("nan"))
    rnd = random.Random(seed)
    vals = sorted(ece(rnd.choices(ans, k=len(ans)))[0] for _ in range(n_boot))
    return vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1]


def per_class(rows, critical):
    classes = sorted({c for r in rows for c in r["truth"]} | {r["preds"][0][0] for r in rows if answered(r)})
    out = []
    for c in classes:
        pos = [r for r in rows if c in r["truth"]]
        neg = [r for r in rows if c not in r["truth"]]
        tp1 = sum(1 for r in pos if answered(r) and r["preds"][0][0] == c)
        tp3 = sum(1 for r in pos if answered(r) and c in [l for l, _ in r["preds"][:3]])
        fp = sum(1 for r in neg if answered(r) and r["preds"][0][0] == c)
        ab = sum(1 for r in pos if not answered(r))
        wrong_conf = len(pos) - tp3 - ab
        out.append(
            dict(
                cls=c,
                n_pos=len(pos),
                sens1=fmt(tp1, len(pos)),
                sens3=fmt(tp3, len(pos)),
                spec=fmt(len(neg) - fp, len(neg)),
                ppv=fmt(tp1, tp1 + fp),
                abstained=ab,
                confident_miss=wrong_conf,
                critical=c in critical,
            )
        )
    return out


def evaluate(rows, args, label):
    n = len(rows)
    L = [f"\n================ {label}  (n={n}) ================"]
    ab = sum(1 for r in rows if not answered(r))
    k1, k3 = sum(top1_ok(r) for r in rows), sum(top3_ok(r) for r in rows)
    ans = [r for r in rows if answered(r)]
    L.append(f"Top-1 (abstain=wrong): {fmt(k1, n)}")
    L.append(f"Top-3 (abstain=wrong): {fmt(k3, n)}")
    L.append(f"Abstained: {ab}/{n}   |   Top-1 among answered: {fmt(sum(top1_ok(r) for r in ans), len(ans))}")
    if n < 100:
        L.append(f"!! n={n} صغير: أي رقم هنا بفاصل ثقة واسع جداً ولا يصلح لإثبات {100*args.target:.0f}%")

    tab, best = selective_table(rows, args.target)
    L.append("\nSelective accuracy (coverage = share of ALL images answered, sorted by confidence):")
    for cov, i, k, p, lo, hi, thr in tab:
        L.append(f"  coverage {int(cov*100):3d}% (answered {i:4d}, conf>={thr:.2f}): accuracy {100*p:5.1f}% [{100*lo:4.1f}-{100*hi:4.1f}]")
    if best:
        L.append(f"  -> largest coverage with point accuracy >= {100*args.target:.0f}%: {int(best[0]*100)}% (lower CI {100*best[4]:.1f}%)")
    else:
        L.append(f"  -> no coverage level reaches point accuracy {100*args.target:.0f}%")

    vals = [r for r in rows if r["split"] == "val"]
    tests = [r for r in rows if r["split"] == "test"]
    if vals and tests:
        thr = pick_threshold(vals, args.target)
        if thr is None:
            L.append(f"\nThreshold picked on VAL: none reaches {100*args.target:.0f}% -> system cannot meet the target on val")
        else:
            sel = [r for r in tests if answered(r) and r["preds"][0][1] >= thr]
            k = sum(top1_ok(r) for r in sel)
            L.append(f"\nThreshold picked on VAL = {thr:.3f}; applied once to TEST: coverage {fmt(len(sel), len(tests))}")
            L.append(f"  TEST accuracy among answered: {fmt(k, len(sel))}")
    else:
        L.append("\n(لا يوجد split=val/test: منحنى التغطية أعلاه محسوب على نفس الصور، فهو متفائل. اختر الحد على val وقِس على test.)")

    e, rel = ece(rows)
    lo, hi = boot_ece(rows)
    L.append(f"\nCalibration: ECE = {e:.3f}  (95% bootstrap CI {lo:.3f}-{hi:.3f})  — لا تعرض الثقة كنسبة مئوية قبل أن يكون ECE صغيراً بفاصل ضيق")
    for blo, bhi, m, conf, acc in rel:
        L.append(f"  conf {blo:.1f}-{bhi:.1f}: n={m:4d}  mean_conf={conf:.2f}  observed_acc={acc:.2f}")

    L.append("\nPer class (one-vs-rest; abstain counts as a miss for sensitivity):")
    crit = set(args.critical.split(",")) if args.critical else set()
    for d in per_class(rows, crit):
        flag = " [CRITICAL]" if d["critical"] else ""
        warn = "  (n<30!)" if d["n_pos"] < 30 else ""
        L.append(
            f"  {d['cls'][:34]:34s} n={d['n_pos']:3d}{flag}{warn}\n"
            f"      sens@1 {d['sens1']}\n"
            f"      sens@3 {d['sens3']}\n"
            f"      spec   {d['spec']}   PPV {d['ppv']}\n"
            f"      abstained={d['abstained']}  confident_miss={d['confident_miss']}"
        )
    cr = [r for r in rows if r["truth"] & crit]
    if crit and cr:
        conf_miss = sum(1 for r in cr if answered(r) and not (r["truth"] & {l for l, _ in r["preds"][:3]}))
        L.append(f"\nCRITICAL cases: {len(cr)} | confident miss (answered, true critical class not in top-3): {fmt(conf_miss, len(cr))} | abstained: {sum(1 for r in cr if not answered(r))}")

    L.append("\nHow many labeled test images are needed to *demonstrate* the target (95% CI lower bound):")
    for ptrue in (0.96, 0.97, 0.98):
        L.append(f"  if true accuracy is {int(ptrue*100)}%: need ~{n_needed(ptrue, 0.93) or '>20000'} images for lower bound >= 93%, ~{n_needed(ptrue, 0.95) or '>20000'} for >= 95%")
    return "\n".join(L)


def compare(a_rows, b_rows):
    A = {r["file"]: r for r in a_rows}
    B = {r["file"]: r for r in b_rows}
    common = sorted(set(A) & set(B))
    if not common:
        return "\n(no common filenames to compare)"
    b = sum(1 for f in common if top1_ok(A[f]) and not top1_ok(B[f]))
    c = sum(1 for f in common if top1_ok(B[f]) and not top1_ok(A[f]))
    ka, kb = sum(top1_ok(A[f]) for f in common), sum(top1_ok(B[f]) for f in common)
    rnd = random.Random(3)
    diffs = []
    for _ in range(2000):
        s = rnd.choices(common, k=len(common))
        diffs.append((sum(top1_ok(A[f]) for f in s) - sum(top1_ok(B[f]) for f in s)) / len(s))
    diffs.sort()
    return (
        f"\n================ PAIRED COMPARISON (A = --pred, B = --compare), n={len(common)} ================\n"
        f"A top-1: {fmt(ka, len(common))}\nB top-1: {fmt(kb, len(common))}\n"
        f"only A correct = {b}, only B correct = {c}; exact McNemar p = {mcnemar_exact(b, c):.3f}\n"
        f"A - B accuracy difference: {100*(ka-kb)/len(common):+.1f} points  (95% bootstrap CI {100*diffs[50]:+.1f} to {100*diffs[1949]:+.1f})\n"
        f"{'=> الفرق غير دال إحصائياً' if mcnemar_exact(b, c) >= 0.05 else '=> الفرق دال إحصائياً (p<0.05)'}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--compare")
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--critical", default="")
    a = ap.parse_args()
    rows = load(a.pred)
    domains = sorted({r["domain"] for r in rows})
    for d in domains:
        print(evaluate([r for r in rows if r["domain"] == d], a, f"domain = {d}"))
    if len(domains) > 1:
        print(evaluate(rows, a, "ALL domains pooled (للمقارنة فقط؛ الهدف يُقاس لكل نطاق)"))
    if a.compare:
        print(compare(rows, load(a.compare)))


if __name__ == "__main__":
    sys.exit(main())

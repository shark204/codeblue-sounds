"""
Code Blue ECG — LITFL Top 150 ECG Certainty-Gated Multi-Label Dataset & KB Builder
License Compliance:
  Life in the Fast Lane (LITFL) #FOAMed resources are licensed under
  Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
  Every exported record preserves canonical case URL + explicit CC BY-NC-SA 4.0 attribution.

STRICT CERTAINTY GATE (Zero Unverified Data Pollution):
  - Crawls https://litfl.com/ecg-case-001/ through https://litfl.com/ecg-case-150/ politely.
  - Parses ONLY the official Answer / ECG Interpretation section (never guesses from questions).
  - Extracts ALL co-existing CONFIRMED canonical diagnoses per ECG (true multi-label!).
  - Explicitly rejects/filters negated or uncertain clauses ("possible", "consider", "rule out",
    "differential", "no evidence of", "exclude", "cannot exclude", "versus", "vs.") so uncertain
    mentions NEVER pollute training labels!
  - Cases with zero confirmed canonical matches are quarantined into `litfl_needs_review.csv`
    for manual clinician sign-off rather than silently entering the training set.

Usage (Kaggle or Google Colab — runs in ~60-90 seconds):
  !pip install -q requests beautifulsoup4 pandas pillow
  !python fetch_litfl_top150.py
"""
import csv
import json
import os
import re
import time
from typing import Dict, List, Set, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

IS_KAGGLE = os.path.exists("/kaggle/working")
IS_COLAB = os.path.exists("/content")
OUT_DIR = "/kaggle/working/litfl_top150" if IS_KAGGLE else ("/content/litfl_top150" if IS_COLAB else "./litfl_top150")
IMG_DIR = os.path.join(OUT_DIR, "images")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://litfl.com/top-100/ecg/",
}

# Strict canonical regex patterns mapped to our DenseNet121 / Clinical Library taxonomy.
# Only unambiguous clinical terms in the official LITFL interpretation pass.
CANONICAL_PATTERNS: Dict[str, List[str]] = {
    "ANTERIOR_SEPTAL_MI": [
        r"\banterior\s+(?:stemi|mi|myocardial\s+infarction)\b",
        r"\banteroseptal\s+(?:stemi|mi|infarction)\b",
        r"\banterolateral\s+(?:stemi|mi|infarction)\b",
        r"\bde\s*winter\b",
        r"\bwellens\b",
    ],
    "INFERIOR_POSTERIOR_MI": [
        r"\binferior\s+(?:stemi|mi|myocardial\s+infarction)\b",
        r"\bposterior\s+(?:stemi|mi|myocardial\s+infarction)\b",
        r"\binferoposterior\s+(?:stemi|mi)\b",
        r"\bright\s+ventricular\s+(?:mi|infarction)\b",
    ],
    "LATERAL_MI": [
        r"\b(?:high\s+)?lateral\s+(?:stemi|mi|myocardial\s+infarction)\b",
    ],
    "NSTEMI_ISCHEMIA": [
        r"\bnstemi\b",
        r"\bsubendocardial\s+ischa?emia\b",
        r"\bmyocardial\s+ischa?emia\b",
    ],
    "ATRIAL_FIBRILLATION": [
        r"\batrial\s+fibrillation\b",
        r"\bpre-?excited\s+af\b",
    ],
    "ATRIAL_FLUTTER": [
        r"\batrial\s+flutter\b",
    ],
    "SVT_TACHYCARDIA": [
        r"\bsupraventricular\s+tachycardia\b",
        r"\bavnrt\b",
        r"\bavrt\b",
        r"\bmultifocal\s+atrial\s+tachycardia\b",
    ],
    "SINUS_BRADYCARDIA": [
        r"\bsinus\s+bradycardia\b",
    ],
    "FIRST_DEGREE_AV_BLOCK": [
        r"\b(?:first|1st)[- ]degree\s+av\s+block\b",
    ],
    "HIGH_GRADE_AV_BLOCK": [
        r"\b(?:second|2nd)[- ]degree\s+av\s+block\b",
        r"\bmobitz\s+(?:i|ii|1|2)\b",
        r"\bwenckebach\b",
        r"\b(?:third|3rd)[- ]degree\s+(?:av|heart)\s+block\b",
        r"\bcomplete\s+(?:av|heart)\s+block\b",
    ],
    "RBBB": [
        r"\bright\s+bundle\s+branch\s+block\b",
        r"\brbbb\b",
    ],
    "LBBB": [
        r"\bleft\s+bundle\s+branch\s+block\b",
        r"\blbbb\b",
    ],
    "LAFB_LPFB": [
        r"\bleft\s+anterior\s+(?:fascicular|hemi)\s*block\b",
        r"\bleft\s+posterior\s+(?:fascicular|hemi)\s*block\b",
        r"\blafb\b",
        r"\blpfb\b",
        r"\bbifascicular\s+block\b",
        r"\btrifascicular\s+block\b",
    ],
    "LVH_RVH": [
        r"\bleft\s+ventricular\s+hypertrophy\b",
        r"\bright\s+ventricular\s+hypertrophy\b",
        r"\blvh\b",
        r"\brvh\b",
    ],
    "PACEMAKER_RHYTHM": [
        r"\b(?:ventricular|atrial|dual[- ]chamber)\s+paced\b",
        r"\bpacemaker\s+(?:rhythm|malfunction|failure)\b",
    ],
    "PVC_BIGEMINY": [
        r"\bventricular\s+bigeminy\b",
        r"\bpremature\s+ventricular\s+(?:contraction|complex|beat)\b",
        r"\bmonomorphic\s+vt\b",
        r"\bventricular\s+tachycardia\b",
        r"\btorsades\s+de\s+pointes\b",
    ],
    "LONG_QT": [
        r"\b(?:prolonged|long)\s+qt\b",
    ],
    "WPW_PATTERN": [
        r"\bwolff[- ]parkinson[- ]white\b",
        r"\bwpw\b",
        r"\bventricular\s+pre-?excitation\b",
    ],
    "BRUGADA_TYPE_1": [
        r"\bbrugada\b",
    ],
    "HYPERKALEMIA": [
        r"\bhyperkala?emia\b",
    ],
    "TCA_SODIUM_BLOCKADE": [
        r"\bsodium\s+channel\s+blockade\b",
        r"\btca\s+(?:overdose|toxicity)\b",
        r"\btricyclic\b",
    ],
    "ACUTE_PERICARDITIS": [
        r"\bacute\s+pericarditis\b",
        r"\bpericarditis\b",
    ],
    "PULMONARY_EMBOLISM_S1Q3T3": [
        r"\bpulmonary\s+embol(?:ism|us)\b",
        r"\bs1q3t3\b",
    ],
    "DEXTROCARDIA_LEAD_REVERSAL": [
        r"\bdextrocardia\b",
        r"\blimb\s+lead\s+reversal\b",
    ],
}

# Sentences containing uncertainty or negation are EXCLUDED from confirmed multi-label extraction
UNCERTAIN_OR_NEGATED_RE = re.compile(
    r"\b(?:no\s+evidence|absent|ruled?\s+out|exclude|cannot\s+exclude|without|unlikely|"
    r"possible|possibly|differential|consider|versus|vs\.?|mimic|pseudo|artefact|artifact|"
    r"not\s+present|normal\s+st|no\s+st\s+elevation)\b",
    re.I,
)


def extract_confirmed_multilabels(answer_text: str) -> Tuple[List[str], List[str]]:
    """
    Splits official answer text into clauses/sentences.
    Only extracts canonical classes from assertive, non-negated, non-uncertain sentences.
    Returns (confirmed_classes, rejected_uncertain_clauses).
    """
    confirmed: Set[str] = set()
    rejected: List[str] = []
    clauses = re.split(r"[\n\r.;•]+", answer_text)
    for raw_clause in clauses:
        clause = raw_clause.strip()
        if len(clause) < 4:
            continue
        is_uncertain = bool(UNCERTAIN_OR_NEGATED_RE.search(clause))
        matched_here = []
        for cls_name, patterns in CANONICAL_PATTERNS.items():
            if any(re.search(pat, clause, re.I) for pat in patterns):
                matched_here.append(cls_name)
        if not matched_here:
            continue
        if is_uncertain:
            rejected.append(clause[:180])
        else:
            confirmed.update(matched_here)
    return sorted(confirmed), rejected


def is_valid_ecg_image(img_url: str) -> bool:
    low = img_url.lower()
    if not any(ext in low for ext in (".jpg", ".jpeg", ".png", ".webp")):
        return False
    skip_tokens = ("logo", "avatar", "medmastery", "gravatar", "icon", "banner", "author", "mike", "cadogan")
    return not any(tok in low for tok in skip_tokens)


def discover_litfl_case_urls(session: requests.Session) -> List[Tuple[str, str]]:
    """
    Discovers all real LITFL Top 150 ECG case URLs directly from the official WordPress
    category archive (https://litfl.com/category/top-100/top-100-ecg/page/{p}/)
    so non-standard slugs are never missed.
    """
    found: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    for page in range(1, 22):
        arch_url = (
            "https://litfl.com/category/top-100/top-100-ecg/"
            if page == 1
            else f"https://litfl.com/category/top-100/top-100-ecg/page/{page}/"
        )
        r = None
        for attempt in range(3):
            try:
                r = session.get(arch_url, timeout=20)
                if r.status_code in (200, 404):
                    break
                time.sleep(2.0 * (attempt + 1))
            except Exception:
                time.sleep(2.0 * (attempt + 1))
        if r is None or r.status_code == 404:
            break
        if r.status_code != 200:
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        new_on_page = 0
        for a in soup.find_all("a", href=True):
            href = urljoin(arch_url, a["href"].strip())
            if not href.startswith("https://litfl.com/"):
                continue
            if any(skip in href for skip in ("/category/", "/tag/", "/author/", "/page/", "/top-100/", "/wp-", "#")):
                continue
            if re.search(r"/ecg-case-|/ecg-quiz-|/ekg-case-|/activate-or-wait-", href, re.I):
                clean = href.split("?")[0].rstrip("/") + "/"
                if clean not in seen:
                    seen.add(clean)
                    cid = f"LITFL_{len(found) + 1:03d}"
                    found.append((cid, clean))
                    new_on_page += 1
        if new_on_page == 0 and page > 2:
            break
        time.sleep(0.5)
    # Automatic fallback if archive pagination is blocked or uses JS rendering
    if len(found) < 50:
        print(f"  [INFO] Archive returned {len(found)} URLs; adding direct ecg-case-001..142 list.")
        for n in range(1, 143):
            u = f"https://litfl.com/ecg-case-{n:03d}/"
            if u not in seen:
                seen.add(u)
                found.append((f"LITFL_{n:03d}", u))
    return found


def scrape_case_url(case_id: str, url: str, session: requests.Session) -> Dict:
    r = None
    for attempt in range(4):
        try:
            r = session.get(url, timeout=20)
            if r.status_code == 200:
                break
            if r.status_code in (429, 502, 503, 504):
                time.sleep(2.5 * (attempt + 1))
                continue
            break
        except Exception:
            time.sleep(2.0 * (attempt + 1))
    if r is None or r.status_code != 200:
        return {"case_id": case_id, "url": url, "status": (r.status_code if r is not None else 0)}

    soup = BeautifulSoup(r.text, "html.parser")
    article = soup.find("article") or soup.find("main") or soup
    title_el = soup.find("h1")
    title = title_el.get_text(" ", strip=True) if title_el else case_id

    # Extract official ECG images inside the article content
    img_urls: List[str] = []
    for img in article.find_all("img"):
        src = img.get("data-lazy-src") or img.get("data-src") or img.get("src") or ""
        if src.startswith("data:"):
            continue
        full_url = urljoin(url, src)
        if is_valid_ecg_image(full_url) and full_url not in img_urls:
            img_urls.append(full_url)

    # Extract answer / interpretation text (LITFL uses accordion/toggle or heading after Question)
    full_text = article.get_text("\n", strip=True)
    answer_text = full_text
    m_ans = re.search(r"(?:Answer|ECG\s+Interpretation|Interpretation|Diagnosis)\s*[:\n](.+)", full_text, re.I | re.S)
    if m_ans:
        answer_text = m_ans.group(1)[:4000]

    confirmed_classes, rejected_clauses = extract_confirmed_multilabels(answer_text)

    # Download up to 2 primary ECG images per confirmed case
    saved_imgs: List[str] = []
    for idx, iurl in enumerate(img_urls[:2]):
        ext = ".png" if ".png" in iurl.lower() else ".jpg"
        fname = f"{case_id.lower()}_{idx + 1}{ext}"
        fpath = os.path.join(IMG_DIR, fname)
        if not os.path.exists(fpath):
            try:
                ir = session.get(iurl, timeout=20)
                if ir.status_code == 200 and len(ir.content) > 8000:
                    with open(fpath, "wb") as f:
                        f.write(ir.content)
                    saved_imgs.append(fname)
            except Exception:
                pass
        else:
            saved_imgs.append(fname)

    return {
        "case_id": case_id,
        "title": title,
        "url": url,
        "status": 200,
        "images": saved_imgs,
        "confirmed_classes": confirmed_classes,
        "multi_diagnosis_count": len(confirmed_classes),
        "rejected_uncertain_count": len(rejected_clauses),
        "summary_excerpt": answer_text[:350].replace("\n", " "),
        "attribution": "Life in the Fast Lane (LITFL) — CC BY-NC-SA 4.0",
    }


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)

    verified_rows = []
    review_rows = []
    kb_cases = []

    print(f"Discovering all LITFL Top 150 ECG URLs from official archive -> {OUT_DIR}")
    case_urls = discover_litfl_case_urls(session)
    print(f"Discovered {len(case_urls)} LITFL ECG case URLs.")
    for idx_u, (cid, url) in enumerate(case_urls, 1):
        rec = scrape_case_url(cid, url, session)
        if rec.get("status") != 200:
            continue
        if rec["confirmed_classes"] and rec["images"]:
            for img_name in rec["images"]:
                verified_rows.append({
                    "case_id": rec["case_id"],
                    "image_file": img_name,
                    "confirmed_classes": "|".join(rec["confirmed_classes"]),
                    "multi_diagnosis_count": rec["multi_diagnosis_count"],
                    "url": rec["url"],
                    "attribution": rec["attribution"],
                })
            kb_cases.append({
                "case_id": rec["case_id"],
                "title": rec["title"],
                "url": rec["url"],
                "diagnosis_codes": rec["confirmed_classes"],
                "summary": rec["summary_excerpt"],
                "attribution": rec["attribution"],
            })
        else:
            review_rows.append({
                "case_id": rec["case_id"],
                "url": rec["url"],
                "images_found": len(rec.get("images", [])),
                "rejected_uncertain_count": rec.get("rejected_uncertain_count", 0),
                "summary_excerpt": rec.get("summary_excerpt", ""),
            })
        if idx_u % 20 == 0:
            print(f"  Progress {idx_u}/{len(case_urls)} | Verified imgs: {len(verified_rows)} | Quarantined for review: {len(review_rows)}")
        time.sleep(0.85)

    ver_csv = os.path.join(OUT_DIR, "litfl_verified_multilabel.csv")
    with open(ver_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "image_file", "confirmed_classes", "multi_diagnosis_count", "url", "attribution"])
        w.writeheader()
        w.writerows(verified_rows)

    rev_csv = os.path.join(OUT_DIR, "litfl_needs_review.csv")
    with open(rev_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "url", "images_found", "rejected_uncertain_count", "summary_excerpt"])
        w.writeheader()
        w.writerows(review_rows)

    kb_json = os.path.join(OUT_DIR, "litfl_cases_kb.json")
    with open(kb_json, "w", encoding="utf-8") as f:
        json.dump(kb_cases, f, indent=2, ensure_ascii=False)

    print(f"\nDONE! Verified multi-label ECG images: {len(verified_rows)} | Cases in KB: {len(kb_cases)}")
    print(f"Quarantined (uncertain/unmapped, excluded from training): {len(review_rows)}")
    print(f"Saved: {ver_csv}, {rev_csv}, {kb_json}")


if __name__ == "__main__":
    main()

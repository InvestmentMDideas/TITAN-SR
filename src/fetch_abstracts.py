"""Reconstruct title + abstract from PubMed for a TITAN-SR label manifest, producing a training/eval-ready CSV
(review_id, criteria, title, abstract, label). Abstracts are fetched, not redistributed.

Usage:
  python src/fetch_abstracts.py --labels data/manifests/pmc_train_labels.csv.gz \
         --criteria data/manifests/criteria.csv.gz --out data/pmc_train.csv
Set NCBI_API_KEY in the environment for a higher rate limit (optional).
"""
import argparse, csv, gzip, os, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
csv.field_size_limit(10**9)

def opn(p):
    return gzip.open(p, "rt", encoding="utf-8", errors="replace") if p.endswith(".gz") else open(p, encoding="utf-8", errors="replace")

def efetch(pmids, key):
    data = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(pmids), "rettype": "abstract",
                                   "retmode": "xml", **({"api_key": key} if key else {})}).encode()
    req = urllib.request.Request("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", data=data)
    for _ in range(5):
        try:
            return urllib.request.urlopen(req, timeout=90).read()
        except Exception:
            time.sleep(1.5)
    return None

def parse(xmlb):
    out = {}
    try: root = ET.fromstring(xmlb)
    except Exception: return out
    for art in root.iter("PubmedArticle"):
        pid = art.find(".//MedlineCitation/PMID")
        if pid is None: continue
        ti = art.find(".//Article/ArticleTitle")
        title = "".join(ti.itertext()).strip() if ti is not None else ""
        parts = []
        for ab in art.findall(".//Article/Abstract/AbstractText"):
            lbl = ab.get("Label"); txt = "".join(ab.itertext()).strip()
            if txt: parts.append(f"{lbl}: {txt}" if lbl else txt)
        out[pid.text] = (title, " ".join(parts))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True); ap.add_argument("--criteria", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=200); a = ap.parse_args()
    key = os.environ.get("NCBI_API_KEY")
    crit = {}
    with opn(a.criteria) as f:
        for r in csv.DictReader(f): crit[r["review_id"]] = r.get("eligibility_criteria") or r.get("selection_criteria") or ""
    rows = [r for r in csv.DictReader(opn(a.labels))]
    pmids = sorted({r["candidate_pmid"] for r in rows if (r.get("candidate_pmid") or "").strip().isdigit()})
    cache = {}
    for i in range(0, len(pmids), a.batch):
        b = pmids[i:i+a.batch]; xmlb = efetch(b, key)
        if xmlb: cache.update(parse(xmlb))
        if (i // a.batch) % 20 == 0: print(f"  fetched {i+len(b)}/{len(pmids)}", flush=True)
        time.sleep(0.12)
    with open(a.out, "w", encoding="utf-8", newline="") as g:
        w = csv.writer(g); w.writerow(["review_id", "criteria", "title", "abstract", "label"]); n = 0
        for r in rows:
            pm = (r.get("candidate_pmid") or "").strip()
            ti, ab = cache.get(pm, ("", ""))
            if not ab: continue
            w.writerow([r["review_id"], crit.get(r["review_id"], ""), ti, ab, r["label"]]); n += 1
    print(f"wrote {n} rows -> {a.out}")

if __name__ == "__main__":
    main()

"""Stateless commercial-chatbot screening benchmark (one API request per citation; no conversation state).
API keys are read from environment variables (OPENAI_API_KEY, DEEPSEEK_API_KEY, ...). No keys are shipped."""
import argparse, csv, json, os, re, time, urllib.request
csv.field_size_limit(10**9)
SYS = ("You are an expert systematic-review screener. Decide whether a candidate citation should be included "
       "for full-text review based on the eligibility criteria. Output ONLY JSON: "
       '{"decision":"INCLUDE"|"EXCLUDE","confidence":0.0-1.0}.')
ENDPOINTS = {  # provider -> (url, model, env_var)
    "openai":  ("https://api.openai.com/v1/chat/completions", "gpt-4o-2024-11-20", "OPENAI_API_KEY"),
    "deepseek":("https://api.deepseek.com/chat/completions",  "deepseek-chat",     "DEEPSEEK_API_KEY"),
}
def call(provider, criteria, title, abstract):
    url, model, env = ENDPOINTS[provider]; key = os.environ.get(env)
    if not key: raise SystemExit(f"set {env}")
    user = f"ELIGIBILITY CRITERIA:\n{criteria[:1500]}\n\nCANDIDATE:\nTitle: {title}\nAbstract: {abstract[:1500]}\n\nJSON only."
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": 50,
                       "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(url, data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for _ in range(4):
        try:
            c = json.load(urllib.request.urlopen(req, timeout=60))["choices"][0]["message"]["content"]
            m = re.search(r"\{[^{}]*\}", c)
            if m:
                d = json.loads(m.group(0)); return str(d.get("decision","")).upper(), float(d.get("confidence", 0.5))
        except Exception: time.sleep(2)
    return "?", 0.5
def rank_score(dec, conf):  # decision-adjusted ranking score in [0,1]
    return conf if dec == "INCLUDE" else (1 - conf if dec == "EXCLUDE" else 0.5)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--provider", required=True, choices=list(ENDPOINTS))
    ap.add_argument("--input", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cols", default="review_id,criteria,title,abstract,label"); a = ap.parse_args()
    rid, crit, title, abs_, lab = a.cols.split(","); reviews = {}
    with open(a.input, encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            d = reviews.setdefault(r[rid], {"criteria": r.get(crit,"") or "", "rows": []})
            d["rows"].append((r.get(title,"") or "", r.get(abs_,"") or "", int(float(r.get(lab,0) or 0))))
    res = {}
    for k, d in reviews.items():
        labels, scores = [], []
        for t, ab, l in d["rows"]:
            dec, conf = call(a.provider, d["criteria"], t, ab)
            labels.append(l); scores.append(rank_score(dec, conf))
        res[k] = {"labels": labels, "score": scores}
    json.dump(res, open(a.out, "w")); print(f"scored {len(res)} reviews -> {a.out} (evaluate.py for metrics)")
if __name__ == "__main__":
    main()

"""Fine-tune one biomedical BERT checkpoint with the TITAN-SR recipe.

Input format per citation: [CLS] eligibility_criteria [SEP] "title. abstract" [SEP], truncated to 512 tokens,
[CLS] -> single linear head -> sigmoid inclusion probability.
Recipe: 10:1 false-negative-weighted BCE loss + class-balanced WeightedRandomSampler; AdamW lr 2e-5,
effective batch size 128 (per-device 64 x grad-accum 2); up to 10 epochs with early stopping on validation AUC.
"""
import argparse, csv
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score
csv.field_size_limit(10**9)

def load(path, cmap):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            lab = str(r.get(cmap["label"], "")).strip()
            if lab not in ("0", "1", "0.0", "1.0"): continue
            crit = r.get(cmap["criteria"], "") or ""
            text = f'{r.get(cmap["title"],"") or ""}. {r.get(cmap["abstract"],"") or ""}'.strip()
            rows.append((crit, text, int(float(lab))))
    return rows

class DS(Dataset):
    def __init__(s, rows, tok, maxlen): s.r, s.tok, s.maxlen = rows, tok, maxlen
    def __len__(s): return len(s.r)
    def __getitem__(s, i):
        c, t, l = s.r[i]
        e = s.tok(c, t, truncation=True, max_length=s.maxlen, padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in e.items()}, torch.tensor(float(l))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="HF model id, e.g. microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    ap.add_argument("--train", required=True); ap.add_argument("--val", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--maxlen", type=int, default=512); ap.add_argument("--per_device_bs", type=int, default=64)
    ap.add_argument("--grad_accum", type=int, default=2); ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_epochs", type=int, default=10); ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--pos_weight", type=float, default=10.0); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cols", default="criteria,title,abstract,label")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    cc = a.cols.split(","); cmap = dict(criteria=cc[0], title=cc[1], abstract=cc[2], label=cc[3])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForSequenceClassification.from_pretrained(a.base, num_labels=1).to(dev)
    TR, VL = load(a.train, cmap), load(a.val, cmap)
    labels = np.array([l for _, _, l in TR])
    cw = np.where(labels == 1, 1.0 / max(labels.sum(), 1), 1.0 / max((labels == 0).sum(), 1))
    sampler = WeightedRandomSampler(torch.tensor(cw, dtype=torch.double), len(cw), replacement=True)
    dl = DataLoader(DS(TR, tok, a.maxlen), batch_size=a.per_device_bs, sampler=sampler, num_workers=4)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([a.pos_weight]).to(dev))
    scaler = torch.amp.GradScaler("cuda") if dev == "cuda" else None
    total = a.max_epochs * (len(dl) // a.grad_accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / max(1, int(0.1 * total))))
    best_auc, best_state, since = -1, None, 0
    for ep in range(a.max_epochs):
        model.train(); opt.zero_grad()
        for i, (x, y) in enumerate(dl):
            x = {k: v.to(dev) for k, v in x.items()}; y = y.to(dev)
            with torch.amp.autocast("cuda", enabled=dev == "cuda"):
                logit = model(**x).logits.squeeze(-1); loss = lossf(logit, y) / a.grad_accum
            (scaler.scale(loss).backward() if scaler else loss.backward())
            if (i + 1) % a.grad_accum == 0:
                if scaler: scaler.step(opt); scaler.update()
                else: opt.step()
                sched.step(); opt.zero_grad()
        # validation AUC
        model.eval(); vs, vy = [], []
        with torch.no_grad():
            for x, y in DataLoader(DS(VL, tok, a.maxlen), batch_size=128):
                x = {k: v.to(dev) for k, v in x.items()}
                vs += torch.sigmoid(model(**x).logits.squeeze(-1)).float().cpu().tolist(); vy += y.tolist()
        auc = roc_auc_score(vy, vs)
        print(f"epoch {ep} val_auc {auc:.4f}", flush=True)
        if auc > best_auc:
            best_auc, since = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= a.patience: break
    if best_state: model.load_state_dict(best_state)
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    print(f"saved {a.out} (best val AUC {best_auc:.4f})")

if __name__ == "__main__":
    main()

# data_utils_speaker.py — ALL generators now emit label+speaker
# ------------------------------------------------------------
# 1. Every meta dict entry:   {"label": 0/1, "speaker": str}
# 2. AudioDataset returns (wave, label, speaker_idx, key)
# 3. split_dict & helpers reference meta[k]["label"]
# 4. Generator functions for *all* datasets filled in with speaker ID logic
# ------------------------------------------------------------

import os, random, csv, librosa, numpy as np, torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader

# ---------------- Device ----------------

device = (
    "cuda" if torch.cuda.is_available() else (
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)

# ---------------- Speaker fallback util ----------------

def extract_spk(path: str):
    base = os.path.basename(path)
    if "_" in base:
        return base.split("_")[0]
    if "-" in base:
        return base.split("-")[0]
    return "single"
def seed_worker(worker_id):
    """
    Used in generating seed for the worker of torch.utils.data.Dataloader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
# ---------------- EER helpers (unchanged) --------------

def compute_det_curve(tar, non):
    n = tar.size + non.size
    scores = np.concatenate((tar, non))
    labels = np.concatenate((np.ones(tar.size), np.zeros(non.size)))
    idx = np.argsort(scores, kind="mergesort")
    labels = labels[idx]
    tar_cum = np.cumsum(labels)
    non_cum = non.size - (np.arange(1, n+1) - tar_cum)
    frr = np.concatenate(([0], tar_cum / tar.size))
    far = np.concatenate(([1], non_cum / non.size))
    thr = np.concatenate(([scores[idx[0]]-1e-3], scores[idx]))
    return frr, far, thr

def compute_eer(gt, pr):
    tar, non = pr[gt==1], pr[gt==0]
    frr, far, thr = compute_det_curve(tar, non)
    i = np.argmin(np.abs(frr-far))
    return (frr[i]+far[i])/2, thr[i]

# ---------------- Validation ---------------------------

def run_validation(model, fx, loader, sr):
    outs, gts = [], []
    model.eval()
    with torch.no_grad():
        for wav, y, _, _ in tqdm(loader, desc="Valid"):
            inp = fx(wav.numpy(), sampling_rate=sr, return_attention_mask=True,
                     padding_value=0, return_tensors="pt").to(device)
            logits = model(**inp).logits
            outs.extend(logits.softmax(-1)[:,1].cpu().tolist())
            gts.extend(y.tolist())
    auroc = roc_auc_score(gts, outs)
    eer, thr = compute_eer(np.array(gts), np.array(outs))
    preds = (np.array(outs) > thr).astype(int)
    acc  = (preds == np.array(gts)).mean()
    print(f"VAL  Acc={acc:.3f}  AUROC={auroc:.3f}  EER={eer:.3f}")
    return acc, auroc, (eer, thr)

# ---------------- Pad helper ---------------------------

def pad(x, max_len=64000):
    return x[:max_len] if len(x)>=max_len else np.concatenate([x, np.zeros(max_len-len(x))])

# ---------------- AudioDataset -------------------------
class AudioDataset(Dataset):
    spk2idx = {}
    def __init__(self, ids, meta, cut=64000):
        self.ids, self.meta, self.cut = ids, meta, cut
        for k in ids:
            spk = meta[k]["speaker"]
            if spk not in AudioDataset.spk2idx:
                AudioDataset.spk2idx[spk] = len(AudioDataset.spk2idx)
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        key = self.ids[idx]
        x,_ = librosa.load(key, sr=8000); x = pad(x, self.cut)
        lab = self.meta[key]["label"]
        spk = AudioDataset.spk2idx[self.meta[key]["speaker"]]
        return x, lab, spk, key

# ---------------- Dataset-specific generators ----------

def genIn_the_wild_list_new(base):
    meta, lst = {}, []
    with open(os.path.join(base, "meta.csv")) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            key = os.path.join(base, r["file"])
            lab = 1 if r["label"]=="bona-fide" else 0
            spk = r.get("speaker", extract_spk(key))
            meta[key] = {"label": lab, "speaker": spk}; lst.append(key)
    return meta, lst

def getASVSpoof2021_list_new(base):
    meta, lst = {}, []
    proto = os.path.join(base, "ASVspoof2021.LA.cm.eval.trl_updated.txt")
    with open(proto) as f:
        for line in f:
            _, uid, spk, _, lab = line.strip().split()
            key = os.path.join(base, "flac", uid+".flac")
            meta[key] = {"label": 1 if lab=="bonafide" else 0, "speaker": spk}; lst.append(key)
    return meta, lst

def getASVSpoof2019_list_new(base):
    meta, lst = {}, []
    protocols = [
        ("ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trl.txt", "ASVspoof2019_LA_train/flac"),
        ("ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.val.trl.txt",   "ASVspoof2019_LA_val/flac"),
        ("ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.test.trl.txt",  "ASVspoof2019_LA_test/flac")]
    for proto, folder in protocols:
        with open(os.path.join(base, proto)) as f:
            for line in f:
                _, uid, spk, _, lab = line.strip().split()
                key = os.path.join(base, folder, uid+".flac")
                meta[key] = {"label": 1 if lab=="bonafide" else 0, "speaker": spk}; lst.append(key)
    return meta, lst

def genLJSpeech_list_new(base):
    meta, lst = {}, []
    for f in os.listdir(os.path.join(base, "wavs")):
        key = os.path.join(base, "wavs", f)
        meta[key] = {"label": 1, "speaker": "LJ"}; lst.append(key)
    return meta, lst

def genWavefake_list_new(base):
    folders = ["ljspeech_melgan", "ljspeech_parallel_wavegan", "ljspeech_multi_band_melgan",
               "ljspeech_full_band_melgan", "ljspeech_waveglow", "ljspeech_hifiGAN"]
    meta, lst = {}, []
    for fld in folders:
        for f in os.listdir(os.path.join(base, fld)):
            key = os.path.join(base, fld, f)
            spk = extract_spk(f)  # p225
            meta[key] = {"label": 0, "speaker": spk}; lst.append(key)
    return meta, lst

def gen_for_norm_list_new(base):
    def read_split(split):
        meta, ids = {}, []
        for cat in ["fake", "real"]:
            folder = os.path.join(base, split, cat)
            for f in os.listdir(folder):
                key = os.path.join(folder, f)
                lab = 0 if cat=="fake" else 1
                spk = extract_spk(f)
                meta[key] = {"label": lab, "speaker": spk}; ids.append(key)
        return meta, ids
    tr_meta,tr_ids = read_split("training")
    va_meta,va_ids = read_split("validation")
    te_meta,te_ids = read_split("testing")
    return tr_ids,tr_meta,va_ids,va_meta,te_ids,te_meta

# ------------------------------------------------------------
# split_dict updated to read meta[label]
# ------------------------------------------------------------

def split_dict(meta, tr=0.1, va=0.1, te=0.8, seed=None):
    if seed is not None:
        random.seed(seed)
    pos = [k for k,v in meta.items() if v["label"]==1]
    neg = [k for k,v in meta.items() if v["label"]==0]
    random.shuffle(pos); random.shuffle(neg)
    def split(lst):
        n=len(lst); a=int(tr*n); b=int(va*n)
        return lst[:a], lst[a:a+b], lst[a+b:]
    tr1,va1,te1 = split(pos); tr0,va0,te0 = split(neg)
    tr_ids, va_ids, te_ids = tr1+tr0, va1+va0, te1+te0
    tr_meta = {k:meta[k] for k in tr_ids}
    va_meta = {k:meta[k] for k in va_ids}
    te_meta = {k:meta[k] for k in te_ids}
    return tr_meta, va_meta, te_meta

# ------------------------------------------------------------
# combine_all_dicts – merges speaker-aware metas
# ------------------------------------------------------------

def combine_all_dicts(meta, tr_lab, tr_ids, va_lab, va_ids, te_lab, te_ids, seed=None):
    tr_m, va_m, te_m = split_dict(meta, seed=seed)
    for dic, store_ids, store_meta in [(tr_m,tr_ids,tr_lab), (va_m,va_ids,va_lab), (te_m,te_ids,te_lab)]:
        store_ids += list(dic.keys()); store_meta.update(dic)
    return tr_lab, tr_ids, va_lab, va_ids, te_lab, te_ids

# ------------------------------------------------------------
# gen_combined_list – unchanged interface, uses new metas
# ------------------------------------------------------------

def gen_combined_list(paths, seed=None):
    tr_lab, va_lab, te_lab = {}, {}, {}
    tr_ids, va_ids, te_ids = [], [], []
    for p in paths:
        if p=="../data/in_the_wild/":
            meta,_=genIn_the_wild_list_new(p)
        elif p=="../data/ASVspoof2021_LA_eval/":
            meta,_=getASVSpoof2021_list_new(p)
        elif p=="../data/wavefake/":
            meta,_=genWavefake_list_new(p)
        elif p=="../data/LJSpeech-1.1/":
            meta,_=genLJSpeech_list_new(p)
        elif p=="../data/for-norm/":
            tr_ids_,tr_meta,va_ids_,va_meta,te_ids_,te_meta = gen_for_norm_list_new(p)
            tr_ids += tr_ids_; va_ids += va_ids_; te_ids += te_ids_
            tr_lab.update(tr_meta); va_lab.update(va_meta); te_lab.update(te_meta); continue
        elif p=="../data/for-rerecorded/":
            tr_ids_,tr_meta,va_ids_,va_meta,te_ids_,te_meta = gen_for_norm_list_new(p)
            tr_ids += tr_ids_; va_ids += va_ids_; te_ids += te_ids_
            tr_lab.update(tr_meta); va_lab.update(va_meta); te_lab.update(te_meta); continue
        elif p=="../data/ASVspoof2019_LA/":
            meta,_=getASVSpoof2019_list_new(p)
        else:
            print(f"Unknown path {p}"); continue
        tr_lab,tr_ids,va_lab,va_ids,te_lab,te_ids = combine_all_dicts(
            meta, tr_lab,tr_ids,va_lab,va_ids,te_lab,te_ids, seed)
    return tr_lab, tr_ids, va_lab, va_ids, te_lab, te_ids

# ------------------------------------------------------------
# Loader helpers – same API
# ------------------------------------------------------------
# ------------------------------------------------------------------
#  Test-only loader (speaker aware, same signature as before)
# ------------------------------------------------------------------
def get_test_loader(data_paths, seed: int, batch_size: int):
    """
    `data_paths` may be a single path (str) or list/tuple of paths.
    Returns a DataLoader that yields (wave, label, speaker_idx, key).
    """
    if isinstance(data_paths, str):
        data_paths = [data_paths]

    test_meta, test_ids = {}, []
    for p in data_paths:
        if p == "../data/in_the_wild/":
            m, ids = genIn_the_wild_list_new(p)
        elif p == "../data/ASVspoof2021_LA_eval/":
            m, ids = getASVSpoof2021_list_new(p)
        elif p == "../data/wavefake/":
            m, ids = genWavefake_list_new(p)
        elif p == "../data/LJSpeech-1.1/":
            m, ids = genLJSpeech_list_new(p)
        elif p == "../data/for-norm/":
            _, _, _, _, ids, m = gen_for_norm_list_new(p)
        elif p == "../data/for-rerecorded/":
            _, _, _, _, ids, m = gen_for_norm_list_new(p)
        elif p == "../data/ASVspoof2019_LA/":
            m, ids = getASVSpoof2019_list_new(p)
        else:
            print(f"[WARN] Unknown path {p} – skipped");  continue

        test_meta.update(m)
        test_ids.extend(ids)

    gen = torch.Generator().manual_seed(seed)
    test_ds = AudioDataset(test_ids, test_meta)
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=gen,
    )
def get_combined_loader(paths, seed, bs):
    tr_lab,tr_ids,va_lab,va_ids,te_lab,te_ids = gen_combined_list(paths, seed)
    g = torch.Generator().manual_seed(seed)
    tr_ds = AudioDataset(tr_ids, tr_lab); va_ds = AudioDataset(va_ids, va_lab); te_ds = AudioDataset(te_ids, te_lab)
    tr_lo = DataLoader(tr_ds, bs, shuffle=True, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    va_lo = DataLoader(va_ds, bs, shuffle=True, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    return tr_lo, va_lo, te_ds
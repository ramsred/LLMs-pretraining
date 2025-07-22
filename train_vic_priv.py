
#!/usr/bin/env python3
# ============================================================
# Wav2Vec2-BERT  •  Spoof-vs-Real  •  Privacy Toolkit
#   · Gradient-Reversal speaker adversary (--use_grl)
#   · Speaker-Null projection              (--null_rank)
#   · Variational Information Bottleneck   (--use_vib)
#   · VICReg projector & loss              (--use_vicreg)
#   · Automatic Mixed Precision            (--amp_dtype fp16|bf16)
# ============================================================

import argparse, os, random, math
from datetime import datetime

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Function
from tqdm import tqdm
from transformers import AutoFeatureExtractor
from sklearn.metrics import roc_auc_score

from models import Wav2Vec2BERT
from data_utils_speaker import (
    get_combined_loader, get_test_loader, run_validation, AudioDataset
)

# ---------------- Device ----------------
DEVICE = ("cuda" if torch.cuda.is_available() else
          "mps"  if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                 else "cpu")

# ---------------- GRL -------------------
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lamb): ctx.lamb = lamb; return x
    @staticmethod
    def backward(ctx, g):      return -ctx.lamb * g, None
def grad_reverse(x, lamb):     return GradReverse.apply(x, lamb)

# ---------------- VIB -------------------
class VIB(nn.Module):
    def __init__(self, d_in: int, d_lat: int = 256, std_floor: float = 0.0):
        super().__init__()
        self.mu     = nn.Linear(d_in, d_lat)
        self.logvar = nn.Linear(d_in, d_lat)
        self.std_floor = std_floor

    def forward(self, h, *, train: bool):
        mu, logvar = self.mu(h), self.logvar(h)

        std = torch.exp(0.5 * logvar)
        if self.std_floor > 0:
            std = torch.clamp(std, min=self.std_floor)
            logvar = 2.0 * torch.log(std + 1e-8)

        eps = torch.randn_like(std) if train else 0.0
        z   = mu + eps * std
        kl  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return z, kl

# ---------------- VICReg ----------------
class VicProjector(nn.Module):
    def __init__(self, d_in, d_lat=256):
        super().__init__()
        self.proj = nn.Linear(d_in, d_lat)
    def forward(self, x): return self.proj(x)

def vicreg_loss(z1, z2, alpha=25.0, beta=25.0, gamma=1.0):
    l_inv = F.mse_loss(z1, z2)

    def _var(z):
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        return torch.mean(F.relu(gamma - std))
    l_var = _var(z1) + _var(z2)

    B, D = z1.shape
    zc = z1 - z1.mean(dim=0)
    cov = (zc.T @ zc) / (B - 1)
    off = cov - torch.diag(torch.diag(cov))
    l_cov = (off ** 2).sum() / D

    return alpha * l_inv + beta * l_var + l_cov

# ---------------- Speaker head ----------
class SpkHead(nn.Module):
    def __init__(self, dim, n_spk):
        super().__init__(); act = nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(dim, 768), act, nn.Dropout(0.2),
            nn.Linear(768, 512), act, nn.Dropout(0.2),
            nn.Linear(512, 256), act, nn.Dropout(0.2),
            nn.Linear(256, 128), act, nn.Dropout(0.2),
            nn.Linear(128, n_spk)
        )
    def forward(self, x): return self.net(x)

# ---------------- Main Model ------------
class PrivacyNet(nn.Module):
    def __init__(self, base, n_spk,
                 *,  # flags
                 use_grl=False,
                 use_vib=False, vib_lat=256, vib_std_floor=0.0,
                 use_vicreg=False, vic_lat=256,
                 null_rank=0):
        super().__init__()
        if use_vib and use_vicreg:
            raise ValueError("Choose either VIB or VICReg, not both.")
        self.base = base
        self.use_grl   = use_grl
        self.use_vib   = use_vib
        self.use_vicreg = use_vicreg
        self.null_rank = null_rank
        self.lamb = 0.0          # set each epoch
        self.beta_vib = 0.0      # set from args

        d_enc = base.config.hidden_size
        d_cls = d_enc  # dim fed to spoof + speaker by default

        # ---- optional VIB ----
        if use_vib:
            self.vib = VIB(d_enc, vib_lat, std_floor=vib_std_floor)
            d_cls = vib_lat

        # ---- optional VICReg ---
        if use_vicreg:
            self.vic_proj = VicProjector(d_enc, vic_lat)
            d_cls = vic_lat  # spoof sees vic latent

        # ---- heads -------------
        self.spoof_head = nn.Linear(d_cls, 2)
        if use_grl:
            self.spk_head  = SpkHead(d_cls, n_spk)

        # ---- speaker-null ------
        if null_rank:
            self.register_buffer("U", torch.zeros(d_enc, null_rank))
            self.U_ok = False

    # null projection
    def _null(self, e): return e - e @ self.U @ self.U.T if self.null_rank and self.U_ok else e

    def forward(self, feats, *, train=True):
        """
        feats: dict from HF extractor.
            If VICReg enabled **during training**, an extra key "twin_feats" must
            carry the second augmented view.
        """
        # ---- main view ----
        h = self.base.wav2vec2bert(
                feats["input_features"],
                attention_mask=feats.get("attention_mask"),
                output_hidden_states=True
            ).last_hidden_state
        emb = self._null(self.base.merged_strategy(h))      # pooled [B,1024]

        # -- VIB --
        kl = torch.zeros(emb.size(0), device=emb.device)
        if self.use_vib:
            emb, kl = self.vib(emb, train=train)

        # -- VICReg --
        vic_z1 = vic_z2 = None
        if self.use_vicreg:
            vic_z1 = self.vic_proj(emb)
            if train and feats.get("twin_feats") is not None:
                twin = feats["twin_feats"]
                h2 = self.base.wav2vec2bert(
                        twin["input_features"],
                        attention_mask=twin.get("attention_mask"),
                        output_hidden_states=True
                    ).last_hidden_state
                emb2   = self._null(self.base.merged_strategy(h2))
                vic_z2 = self.vic_proj(emb2)
            emb = vic_z1                                # classifier input

        # spoof path
        spoof_logits = self.spoof_head(emb)

        # speaker adversary
        spk_logits = None
        if self.use_grl:
            spk_logits = self.spk_head(grad_reverse(emb, self.lamb))

        return spoof_logits, spk_logits, kl, vic_z1, vic_z2

# ---------------- Speaker-Null builder ---
@torch.no_grad()
def update_null(model: PrivacyNet, loader, fx, sr, rank):
    feats, ids = [], []
    for wav, y, spk, _ in loader:
        if not (y == 1).any(): continue
        w = fx(wav[y == 1].float().numpy(), sampling_rate=sr,
               return_attention_mask=True, padding_value=0, return_tensors="pt").to(DEVICE)
        h = model.base.wav2vec2bert(w.input_features, attention_mask=w.attention_mask,
                                    output_hidden_states=True).last_hidden_state
        feats.append(model.base.merged_strategy(h).cpu())
        ids += spk[y == 1].tolist()
    X, ids = torch.cat(feats), torch.tensor(ids)
    mu = X.mean(0, keepdim=True)
    C = sum(((X[ids == s].mean(0, keepdim=True) - mu).T) @
            (X[ids == s].mean(0, keepdim=True) - mu) for s in ids.unique())
    _, V = torch.linalg.eigh(C)
    model.U.copy_(V[:, -rank:]); model.U_ok = True

# ---------------- AMP helper -------------
def amp_context(dtype):
    if dtype == "none":
        from contextlib import nullcontext; return nullcontext(), None
    d = torch.float16 if dtype == "fp16" else torch.bfloat16
    if DEVICE == "cuda":
        ctx = lambda: torch.cuda.amp.autocast(dtype=d)
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "fp16"))
    else:
        ctx = lambda: torch.autocast(device_type=DEVICE, dtype=d)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
    return ctx, scaler

# ---------------- Simple wav augment -----
def weak_augment(wav: torch.Tensor):
    """Label-preserving: ±1% tempo or +0.5 dB noise"""
    if random.random() < 0.5:
        factor = random.uniform(0.99, 1.01)
        wav = torchaudio.functional.time_stretch(wav.unsqueeze(0), 16000, factor).squeeze(0)
    if random.random() < 0.5:
        noise = torch.randn_like(wav) * 0.003
        wav = wav + noise
    return wav

# ---------------- Train / Test -----------
def main(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.output_dir, exist_ok=True)

    tr_loader, val_loader, _ = get_combined_loader(a.data_path, a.seed, a.batch_size)
    n_spk = len(AudioDataset.spk2idx)

    base = Wav2Vec2BERT("facebook/w2v-bert-2.0").to(DEVICE)
    model = PrivacyNet(base, n_spk,
                       use_grl=a.use_grl,
                       use_vib=a.use_vib, vib_lat=a.vib_lat, vib_std_floor=a.vib_std_floor,
                       use_vicreg=a.use_vicreg, vic_lat=a.vic_lat,
                       null_rank=a.null_rank).to(DEVICE)
    model.beta_vib = a.beta_vib

    fx = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0"); sr = 16000
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    amp_ctx, scaler = amp_context(a.amp_dtype)

    # ---- Training ----
    if a.train:
        for ep in range(a.epochs):
            model.lamb = a.lambda_schedule[min(ep, len(a.lambda_schedule)-1)]
            model.train(); ep_loss = 0.0
            for wav, y, spk, _ in tqdm(tr_loader, desc=f"ep{ep}"):
                # primary view
                feats1 = fx(wav.float().numpy(), sampling_rate=sr,
                            return_attention_mask=True, padding_value=0, return_tensors="pt")
                feats1 = {k: v.to(DEVICE) for k, v in feats1.items()}

                # second view only if VICReg enabled
                if a.use_vicreg:
                    wav2 = torch.stack([weak_augment(w) for w in wav])  # simple aug
                    twin = fx(wav2.float().numpy(), sampling_rate=sr, return_attention_mask=True,
                              padding_value=0, return_tensors="pt")
                    feats1["twin_feats"] = {k: v.to(DEVICE) for k, v in twin.items()}

                y, spk = y.to(DEVICE), spk.to(DEVICE)

                with amp_ctx():
                    spoof, spk_logits, kl, z1, z2 = model(feats1, train=True)
                    loss = F.cross_entropy(spoof, y)

                    if a.use_vib:
                        loss += model.beta_vib * kl.mean()

                    if a.use_vicreg and z2 is not None:
                        loss += a.beta_vicreg * vicreg_loss(
                            z1, z2, alpha=a.vic_alpha, beta=a.vic_beta, gamma=a.vic_gamma)

                    if a.use_grl and spk_logits is not None and (y == 1).any():
                        loss += F.cross_entropy(spk_logits[y == 1], spk[y == 1])

                opt.zero_grad(set_to_none=True)
                if scaler: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                else:      loss.backward(); opt.step()

                ep_loss += loss.item()

            print(f"ep{ep}  loss={ep_loss/len(tr_loader):.4f}")

            if a.null_rank: update_null(model, tr_loader, fx, sr, a.null_rank)
            eval_acc, eval_auc, eval_eer = run_validation(model.base, fx, val_loader, sr)
            print(f"Epoch {ep}: Val ACC={eval_acc:.4f} AUROC={eval_auc:.4f} EER={eval_eer:.4f}")

            ck = f"{a.dataset_name}_ep{ep}_{datetime.now():%Y%m%d_%H%M%S}.pth"
            torch.save(model.state_dict(), os.path.join(a.output_dir, ck))

    # ---- Testing ----
    if a.test:
        test_loader = get_test_loader(a.data_path, a.seed, a.batch_size)
        model.load_state_dict(torch.load(os.path.join(a.output_dir, a.checkpoint),
                                         map_location=DEVICE), strict=False)
        model.eval(); run_validation(model.base, fx, test_loader, sr)

# ---------------- CLI --------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser("W2v-BERT • Privacy Toolkit")
    p.add_argument("--data_path", nargs="+", required=True)
    p.add_argument("--output_dir", default="./ckpt")
    p.add_argument("--train", type=bool, default=False)
    p.add_argument("--test",  type=bool, default=False)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=5e-5)
    p.add_argument("--eval_steps", type=int, default=2500)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dataset_name", default="dataset")

    # λ schedule
    p.add_argument("--lambda_adv", default="0.5",
                   help="GRL λ schedule: single float or comma list per epoch")

    # toggles
    p.add_argument("--use_grl", type=bool, default=False)
    p.add_argument("--null_rank", type=int, default=0)

    p.add_argument("--amp_dtype", choices=["none","fp16","bf16"], default="none")

    # VIB
    p.add_argument("--use_vib", type=bool, default=False)
    p.add_argument("--vib_lat", type=int, default=256)
    p.add_argument("--beta_vib", type=float, default=2.5e-2)
    p.add_argument("--vib_std_floor", type=float, default=0.02)

    # VICReg
    p.add_argument("--use_vicreg", type=bool, default=False)
    p.add_argument("--vic_lat", type=int, default=256)
    p.add_argument("--beta_vicreg", type=float, default=1.0)
    p.add_argument("--vic_alpha", type=float, default=25.0)
    p.add_argument("--vic_beta",  type=float, default=25.0)
    p.add_argument("--vic_gamma", type=float, default=1.0)

    args = p.parse_args()

    # λ schedule parsing
    try:
        lam_list = [float(x) for x in args.lambda_adv.split(",")]
    except ValueError:
        raise ValueError("--lambda_adv must be float or comma list")
    args.lambda_schedule = lam_list * args.epochs if len(lam_list) == 1 else lam_list
    if len(args.lambda_schedule) != args.epochs:
        raise ValueError("λ schedule length must match --epochs")

    main(args)

#  python train_vic_priv.py --data_path "../data/in_the_wild/" --train True --use_vic True --epochs 1 --batch_size 64 --amp_dtype bf16 --dataset_name "test_priv"

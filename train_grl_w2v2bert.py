# ============================================================
# Wav2Vec2‑BERT  •  Spoof vs Real  •  Privacy Toolkit
#   • Gradient‑Reversal speaker adversary  (--use_grl)
#   • Speaker‑Null projection               (--null_rank)
#   • Variational Information Bottleneck    (--use_vib)
#   • Automatic Mixed Precision             (--amp_dtype fp16|bf16)
# ============================================================

import argparse, os, random
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Function
from tqdm import tqdm
from transformers import AutoFeatureExtractor
from sklearn.metrics import roc_auc_score

from models import Wav2Vec2BERT
from data_utils_speaker import get_combined_loader, get_test_loader, run_validation, AudioDataset

# ---------------- Device ----------------
DEVICE = ('cuda' if torch.cuda.is_available() else
          'mps'  if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()
                 else 'cpu')

# ---------------- GRL -------------------
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lamb): ctx.lamb = lamb; return x
    @staticmethod
    def backward(ctx, g):      return -ctx.lamb * g, None

def grad_reverse(x, lamb):
    return GradReverse.apply(x, lamb)

# ---------------- VIB -------------------
class VIB(nn.Module):
    def __init__(self, d_in: int, k_lat: int = 256):
        super().__init__()
        self.mu     = nn.Linear(d_in, k_lat)
        self.logvar = nn.Linear(d_in, k_lat)
    def forward(self, h, train=True):
        mu, logvar = self.mu(h), self.logvar(h)
        if train:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return z, kl

# ---------------- Heads -----------------
class SpkHead(nn.Module):
    def __init__(self, dim, n_spk):
        super().__init__(); act = nn.ReLU
        self.net = nn.Sequential(
            nn.Linear(dim, 768), act(), nn.Dropout(0.2),
            nn.Linear(768, 512), act(), nn.Dropout(0.2),
            nn.Linear(512, 256), act(), nn.Dropout(0.2),
            nn.Linear(256, 128), act(), nn.Dropout(0.2),
            nn.Linear(128, n_spk)
        )
    def forward(self, x): return self.net(x)

# ---------------- Model -----------------
class PrivacyNet(nn.Module):
    def __init__(self, base, n_spk, *, use_grl, use_vib, vib_lat, null_rank):
        super().__init__(); self.base = base
        self.use_grl, self.use_vib = use_grl, use_vib
        self.lamb = 0.0; self.beta_vib = 0.0
        self.null_rank = null_rank

        dim = base.config.hidden_size
        if use_vib:
            self.vib = VIB(dim, vib_lat); dim = vib_lat
        self.spoof_head = nn.Linear(dim, 2)
        if use_grl:
            self.spk_head = SpkHead(dim, n_spk)
        if null_rank:
            self.register_buffer('U', torch.zeros(base.config.hidden_size, null_rank)); self.U_ok = False

    def _null(self, e):
        return e - e @ self.U @ self.U.T if self.null_rank and self.U_ok else e

    def forward(self, feats, *, train):
        h = self.base.wav2vec2bert(
            feats['input_features'], attention_mask=feats.get('attention_mask'),
            output_hidden_states=True
        ).last_hidden_state
        emb = self._null(self.base.merged_strategy(h))

        kl = torch.zeros(emb.size(0), device=emb.device)
        if self.use_vib:
            emb, kl = self.vib(emb, train=train)
        spoof = self.spoof_head(emb)
        spk = self.spk_head(grad_reverse(emb, self.lamb)) if self.use_grl else None
        return spoof, spk, kl

# ---------------- SNP builder -----------
@torch.no_grad()
def update_null(model, loader, fx, sr, rank):
    feats, ids = [], []
    for wav, y, spk, _ in loader:
        if not (y == 1).any(): continue
        w = fx(wav[y == 1].float().numpy(), sampling_rate=sr, return_attention_mask=True,
               padding_value=0, return_tensors='pt').to(DEVICE)
        h = model.base.wav2vec2bert(w.input_features, attention_mask=w.attention_mask,
                                    output_hidden_states=True).last_hidden_state
        feats.append(model.base.merged_strategy(h).cpu())
        ids += spk[y == 1].tolist()
    X, ids = torch.cat(feats), torch.tensor(ids)
    mu = X.mean(0, keepdim=True)
    C = sum(((X[ids == s].mean(0, keepdim=True) - mu).T) @ (X[ids == s].mean(0, keepdim=True) - mu)
            for s in ids.unique())
    _, V = torch.linalg.eigh(C)
    model.U.copy_(V[:, -rank:]); model.U_ok = True

# ---------------- AMP helpers -----------
def amp_context(dtype):
    if dtype == 'none':
        from contextlib import nullcontext; return nullcontext(), None
    d = torch.float16 if dtype == 'fp16' else torch.bfloat16
    if DEVICE == 'cuda':
        ctx = lambda: torch.cuda.amp.autocast(dtype=d)
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'fp16'))
    else:
        ctx = lambda: torch.autocast(device_type=DEVICE, dtype=d)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
    return ctx, scaler

# ---------------- Train / Test ----------
def main(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.output_dir, exist_ok=True)

    tr_loader, val_loader, _ = get_combined_loader(a.data_path, a.seed, a.batch_size)
    n_spk = len(AudioDataset.spk2idx)

    base = Wav2Vec2BERT('facebook/w2v-bert-2.0').to(DEVICE)
    model = PrivacyNet(base, n_spk,
                       use_grl=a.use_grl,
                       use_vib=a.use_vib, vib_lat=a.vib_lat,
                       null_rank=a.null_rank).to(DEVICE)
    model.beta_vib = a.beta_vib

    fx = AutoFeatureExtractor.from_pretrained('facebook/w2v-bert-2.0'); sr = 16000

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=a.weight_decay)

    amp_ctx, scaler = amp_context(a.amp_dtype)

    # ---------- training ----------
    if a.train:
        for ep in range(a.epochs):
            model.lamb = a.lambda_schedule[min(ep, len(a.lambda_schedule)-1)]
            model.train(); ep_loss = 0.
            for wav, y, spk, _ in tqdm(tr_loader, desc=f'ep{ep}'):
                feats = fx(wav.float().numpy(), sampling_rate=sr, return_attention_mask=True,
                           padding_value=0, return_tensors='pt')
                feats = {k: v.to(DEVICE) for k, v in feats.items()}
                y, spk = y.to(DEVICE), spk.to(DEVICE)

                with amp_ctx():
                    spoof, spk_logits, kl = model(feats, train=True)
                    loss = F.cross_entropy(spoof, y)
                    if model.use_vib:
                        loss += model.beta_vib * kl.mean()
                    if model.use_grl and spk_logits is not None and (y == 1).any():
                        loss += F.cross_entropy(spk_logits[y == 1], spk[y == 1])

                opt.zero_grad(set_to_none=True)
                if scaler:
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                else:
                    loss.backward(); opt.step()
                ep_loss += loss.item()
            print(f'ep{ep}  loss={ep_loss/len(tr_loader):.4f}')
            
            if a.null_rank:
                update_null(model, tr_loader, fx, sr, a.null_rank)
            eval_acc, eval_auroc, eval_eer = run_validation(model.base, fx, val_loader, sr)
            print(f"Epoch {ep}: Final Validation Accuracy: {eval_acc}, Final Validation AUROC: {eval_auroc}, Final Validation EER: {eval_eer}")
            ck = f"{a.dataset_name}_ep{ep}_{datetime.now():%Y%m%d_%H%M%S}.pth"
            torch.save(model.state_dict(), os.path.join(a.output_dir, ck))
    # ---------- testing ----------
    if a.test:
        test_loader = get_test_loader(a.data_path, a.seed, a.batch_size)
        model.load_state_dict(torch.load(os.path.join(a.output_dir, a.checkpoint), map_location=DEVICE))
        model.eval(); run_validation(model.base, fx, test_loader, sr)

# ---------------- CLI -------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser("W2v-BERT  +  GRL / Speaker-Null toggles")
    p.add_argument("--data_path", nargs="+", required=True)
    p.add_argument("--output_dir", default="./ckpt")
    p.add_argument("--train", type=bool, default=False, help="Flag to run training")
    p.add_argument("--test", type=bool, default=False, help="Flag to run test")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr_spk", type=float, default=5e-5,
                   help="LR for speaker head (used only if --use_grl)")
    p.add_argument("--weight_decay", type=float, default=5e-5)
    p.add_argument("--eval_steps", type=int, default=2500)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dataset_name", default="dataset")
      # old single-value flag removed – replaced by schedule
    p.add_argument(
        "--lambda_adv",
        default="0.5",
        help=("GRL λ.  Either a single float (e.g. 0.5) that will be "
            "broadcast to all epochs, or a comma-separated list whose "
            "length equals --epochs, e.g. 0.3,0.7,1,1.5,2"),
    )
    # -------- toggles --------

    p.add_argument("--use_grl", type=bool, default=False, help="Enable Gradient-Reversal speaker adversary")
    p.add_argument("--null_rank", type=int, default=0,
                   help="Rank of speaker-null projection (0 = off)")

    p.add_argument("--amp_dtype", choices=["none","fp16","bf16"], default="none",
                   help="Automatic Mixed Precision type")

    p.add_argument("--use_gc", type=bool, default=False, help="Enable Gradient-checkpoing")
    p.add_argument("--snr_db", type=float, default=0, help="Signal-to-noise ratio for augmentation")
    p.add_argument("--pitch_shift", type=float, default=0, help="Pitch shift factor for augmentation")
    p.add_argument("--tempo_rate", type=float, default=0, help="Tempo adjustment factor for augmentation")
    p.add_argument("--formant_alpha", type=float, default=0, help="Formant adjustment factor for augmentation")
    p.add_argument("--epsilon", type=float, default=0, help="Epsilon for laplace transformation")


    # ---- VIB toggles ----
    p.add_argument('--use_vib', type=bool, default=False, help='Enable VIB bottleneck')
    p.add_argument('--vib_lat', type=int, default=256, help='Latent dim for VIB')
    p.add_argument('--beta_vib', type=float, default=1e-3, help='Weight for KL term')
    args = p.parse_args()

    # --------- parse λ schedule ---------                              # <<< NEW >>>
    try:
        lam_list = [float(x) for x in args.lambda_adv.split(",")]
    except ValueError:
        raise ValueError("--lambda_adv must be float or comma list")

    if len(lam_list) == 1:
        lam_list *= args.epochs            # broadcast
    elif len(lam_list) != args.epochs:
        raise ValueError(
            f"λ schedule length {len(lam_list)} "
            f"does not match --epochs {args.epochs}"
        )

    args.lambda_schedule = lam_list                                           # <<< NEW >>>
    main(args)

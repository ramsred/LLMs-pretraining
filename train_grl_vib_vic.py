"""
Wav2Vec2‑BERT  •  Spoof vs Real  •  Privacy Toolkit (GRL + VICReg)
===============================================================
Adds a plug‑in **VICReg** loss without touching the original GRL / SNP / Noise
logic.  Toggle it with `--use_vicreg`.  Two views per utterance are generated
on‑the‑fly by **pitch‑shift** and **SNR noise**.  When the flag is off nothing
changes vs your previous script.
"""

import argparse, os, random, math, csv, librosa, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Function
from datetime import datetime
from tqdm import tqdm
from transformers import AutoFeatureExtractor
from sklearn.metrics import roc_auc_score

from models import Wav2Vec2BERT
from data_utils_speaker import (
    get_combined_loader, get_test_loader, run_validation, AudioDataset
)

# -----------------------------------------------------------------------------
# Device ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

device = (
    'cuda' if torch.cuda.is_available() else
    'mps'  if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu'
)

# -----------------------------------------------------------------------------
# GRL -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lamb): ctx.lamb = lamb; return x
    @staticmethod
    def backward(ctx, g):      return -ctx.lamb * g, None

def grad_reverse(x, lamb=0.5):
    return GradReverse.apply(x, lamb)

# -----------------------------------------------------------------------------
# VICReg loss ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def off_diagonal(mat: torch.Tensor):
    n, _ = mat.shape
    return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def vicreg_loss(z1: torch.Tensor, z2: torch.Tensor, sim=25., var=1., cov=1.):
    # invariance (similarity)
    sim_loss = F.mse_loss(z1, z2)
    # variance
    eps = 1e-4
    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0) + eps)
    var_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
    # covariance
    N, D = z1.shape
    z1c = z1 - z1.mean(0)
    z2c = z2 - z2.mean(0)
    cov_z1 = off_diagonal((z1c.T @ z1c) / (N - 1))
    cov_z2 = off_diagonal((z2c.T @ z2c) / (N - 1))
    cov_loss = cov_z1.pow(2).mean() + cov_z2.pow(2).mean()
    return sim * sim_loss + var * var_loss + cov * cov_loss

# -----------------------------------------------------------------------------
# Simple twin‑view sampler  (pitch‑shift  vs  SNR‑noise) ------------------------
# -----------------------------------------------------------------------------

import torchaudio
_noise_buf, _ = torchaudio.load(torchaudio.utils.download_asset(
    'https://pytorch-tutorial-assets.s3.amazonaws.com/steam.wav'))
_noise_buf = _noise_buf.mean(0, keepdim=True)  # mono

def _add_noise(wav: torch.Tensor, snr_db: float):
    sig_pow  = wav.pow(2).mean()
    noise    = _noise_buf[:, : wav.size(-1)].to(wav)
    noise_pow = noise.pow(2).mean() + 1e-9
    k = torch.sqrt(sig_pow / (10 ** (snr_db / 10) * noise_pow))
    return wav + k * noise

def get_twin_views(wav_batch: torch.Tensor, sr: int):
    a_list, b_list = [], []
    for w in wav_batch:
        n_step = random.uniform(-2, 2)
        a = torchaudio.functional.pitch_shift(w.unsqueeze(0), sr, n_step)[0]
        snr  = random.uniform(10, 20)
        b = _add_noise(w.unsqueeze(0), snr)[0]
        a_list.append(a); b_list.append(b)
    return torch.stack(a_list), torch.stack(b_list)

# -----------------------------------------------------------------------------
# Speaker head & noise  (unchanged) -------------------------------------------
# -----------------------------------------------------------------------------

class SpkHead(nn.Module):
    def __init__(self, dim, n_spk):
        super().__init__(); act = nn.ReLU
        self.net = nn.Sequential(
            nn.Linear(dim, 768), act(), nn.Dropout(0.2),
            nn.Linear(768, 512), act(), nn.Dropout(0.2),
            nn.Linear(512, 256), act(), nn.Dropout(0.2),
            nn.Linear(256, 128), act(), nn.Dropout(0.2),
            nn.Linear(128, n_spk))
    def forward(self, x): return self.net(x)

class StructuredNoise(nn.Module):
    def __init__(self, p_drop=0.15, sigma=0.02, momentum=0.1, feat_dim=1024):
        super().__init__(); self.p_drop, self.sigma, self.momentum = p_drop, sigma, momentum
        self.register_buffer('running_std', torch.ones(feat_dim))
    def forward(self, x):
        if not self.training: return x
        if self.p_drop > 0:
            m = torch.rand_like(x) > self.p_drop; x = x * m
        if self.sigma > 0:
            self.running_std = (1 - self.momentum) * self.running_std + self.momentum * x.detach().std(0)
            x = x + torch.randn_like(x) * self.running_std * self.sigma
        return x

# -----------------------------------------------------------------------------
# Wrapper model ---------------------------------------------------------------
# -----------------------------------------------------------------------------

class WAV2V_PRIV(nn.Module):
    def __init__(self, base: Wav2Vec2BERT, n_spk: int, *, use_grl, lamb, null_rank):
        super().__init__(); self.base = base
        H = base.config.hidden_size
        self.use_grl, self.lamb, self.null_rank = use_grl, lamb, null_rank
        self.noise = StructuredNoise(p_drop=0.15, sigma=0.02, feat_dim=2*H)
        if use_grl:
            self.spk_head = SpkHead(2*H, n_spk)
        if null_rank:
            self.register_buffer('U', torch.zeros(H, null_rank)); self.U_ok=False
    def _null(self, e):
        return e - e @ self.U @ self.U.T if self.null_rank and self.U_ok else e
    def forward(self, feats, *, ret_emb=False):
        h = self.base.wav2vec2bert(feats['input_features'], attention_mask=feats.get('attention_mask'),
                                   output_hidden_states=True).last_hidden_state
        mean = self.base.merged_strategy(h)
        mean = self._null(mean)
        spoof_logits = self.base.classifier(mean)
        spk_logits = None
        if self.use_grl:
            std = h.std(1, unbiased=False)
            vec = torch.cat([mean, std], -1)
            vec = self.noise(vec)
            spk_logits = self.spk_head(grad_reverse(vec, self.lamb))
        if ret_emb:
            return spoof_logits, spk_logits, mean
        return spoof_logits, spk_logits

# -----------------------------------------------------------------------------
# Null‑space builder (unchanged) ----------------------------------------------
# -----------------------------------------------------------------------------

@torch.no_grad()
def update_null(model: WAV2V_PRIV, loader, fx, sr, rank):
    xs, spks = [], []
    for wav,y,spk,_ in loader:
        m = y==1;  # bonafide only
        if not m.any(): continue
        ft = fx(wav[m].numpy(), sampling_rate=sr, return_attention_mask=True,
                 padding_value=0, return_tensors='pt').to(device)
        h = model.base.wav2vec2bert(ft.input_features, attention_mask=ft.attention_mask,
                                     output_hidden_states=True).last_hidden_state
        xs.append(model.base.merged_strategy(h).cpu()); spks += spk[m].tolist()
    X = torch.cat(xs); spks = torch.tensor(spks)
    mu = X.mean(0, keepdim=True)
    C = torch.zeros_like(model.U)
    for s in spks.unique():
        d = X[spks==s].mean(0, keepdim=True) - mu
        C += d.T @ d
    _, V = torch.linalg.eigh(C)
    model.U.copy_(V[:,-rank:]); model.U_ok=True

# -----------------------------------------------------------------------------
# Main training ---------------------------------------------------------------
# -----------------------------------------------------------------------------

def main(a):
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.output_dir, exist_ok=True)

    tr_lo, va_lo, _ = get_combined_loader(a.data_path, a.seed, a.batch_size)
    n_spk = len(AudioDataset.spk2idx)
    base  = Wav2Vec2BERT('facebook/w2v-bert-2.0').to(device)
    model = WAV2V_PRIV(base, n_spk, use_grl=a.use_grl, lamb=a.lambda_schedule[0], null_rank=a.null_rank).to(device)
    fx = AutoFeatureExtractor.from_pretrained('facebook/w2v-bert-2.0'); sr=16000

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    # ------------- TRAIN -----------------------------------------------------
    if a.train:
        for ep in range(a.epochs):
            model.train(); model.lamb = a.lambda_schedule[ep]
            ep_loss=0; probs=[]; labels=[]
            for wav,y,spk,_ in tqdm(tr_lo, desc=f'ep{ep}'):
                feats = fx(wav.numpy(), sampling_rate=sr, return_attention_mask=True,
                           padding_value=0, return_tensors='pt')
                feats = {k:v.to(device) for k,v in feats.items()}
                y = y.to(device)

                spoof, spk_logits = model(feats)
                loss = F.cross_entropy(spoof, y)
                if a.use_grl and (y==1).any():
                    loss += F.cross_entropy(spk_logits[y==1], spk[y==1].to(device))

                # VICReg branch (optional)
                if a.use_vicreg:
                    wavA, wavB = get_twin_views(wav, sr)
                    fa = fx(wavA.numpy(), sampling_rate=sr, return_attention_mask=True,
                            padding_value=0, return_tensors='pt'); fa={k:v.to(device) for k,v in fa.items()}
                    fb = fx(wavB.numpy(), sampling_rate=sr, return_attention_mask=True,
                            padding_value=0, return_tensors='pt'); fb={k:v.to(device) for k,v in fb.items()}
                    z1 = model(fa, ret_emb=True)[2]
                    z2 = model(fb, ret_emb=True)[2]
                    loss += vicreg_loss(z1, z2, a.vic_sim, a.vic_var, a.vic_cov)

                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += loss.item(); probs.extend(spoof.softmax(1)[:,1].detach().cpu()); labels.extend(y.cpu())
            print(f"ep{ep} loss={ep_loss/len(tr_lo):.4f} AUROC={roc_auc_score(labels, probs):.3f}")
            update_null(model, tr_lo, fx, sr, a.null_rank) if a.null_rank else None
            run_validation(model.base, fx, va_lo, sr)
            torch.save(model.state_dict(), os.path.join(a.output_dir, f"{a.dataset

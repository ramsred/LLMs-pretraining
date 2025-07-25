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

# ------------------------------------------------------------------
# White-noise helper  (self-contained: no global _noise_buf needed)
# ------------------------------------------------------------------
def _add_noise(wav: torch.Tensor, snr_db: float):
    """Add white noise at a given SNR (dB) to a 1-D torch waveform."""
    sig_pow   = wav.pow(2).mean()
    noise     = torch.randn_like(wav)
    noise_pow = noise.pow(2).mean() + 1e-9
    k = torch.sqrt(sig_pow / (10 ** (snr_db / 10) * noise_pow))
    return wav + k * noise


# ------------------------------------------------------------------
# Twin-view generator:  pitch-shift + SNR noise
# ------------------------------------------------------------------
# def get_twin_views(wav_batch: torch.Tensor, sr: int):
#     """
#     Returns two augmented views of each utterance:
#       • view-A : random pitch-shift (Librosa)
#       • view-B : random additive white noise (SNR 10–20 dB)
#     """
#     a_list, b_list = [], []
#     for w in wav_batch:                                     # w : (T,)
#         # ----- view A : pitch-shift -----------------------
#         n_steps = random.uniform(-2, 2)                     # semitones
#         # 1) torch → numpy (float32, cpu)
#         w_np = w.cpu().numpy().astype(np.float32)
#         # 2) shift
#         shifted_np = librosa.effects.pitch_shift(w_np, sr=sr, n_steps=n_steps)
#         # 3) numpy → torch (keep original device & dtype)
#         a = torch.from_numpy(shifted_np).to(w)

#         # ----- view B : additive noise --------------------
#         snr = random.uniform(10, 20)                        # dB
#         b   = _add_noise(w, snr)

#         a_list.append(a)
#         b_list.append(b)

#     return torch.stack(a_list), torch.stack(b_list)

# ------------------------------------------------------------------
# Twin-view generator: pitch-shift  vs  original
# ------------------------------------------------------------------
def get_twin_views(wav_batch: torch.Tensor, sr: int):
    """
    Returns two views of each utterance:
      • view-A : random pitch-shift (±2 semitones)
      • view-B : the original waveform (identity)
    """
    a_list, b_list = [], []
    for w in wav_batch:                             # w : (T,)
        # ---- view A : pitch-shift --------------------------
        n_steps = random.uniform(-2, 2)             # semitones
        w_np = w.cpu().numpy().astype(np.float32)   # torch → numpy
        shifted_np = librosa.effects.pitch_shift(w_np, sr=8000, n_steps=n_steps)
        a = torch.from_numpy(shifted_np).to(w)      # back to torch

        # ---- view B : original waveform -------------------
        b = w                              # untouched copy

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


# ────────────────── Gradient-accumulation helper ───────────────────
class Accumulator:
    """Simple counter that tells you when to step / zero_grad."""
    def __init__(self, every:int):
        self.every = max(1, every)
    def should_step(self, i):         # i = 0-based mini-batch index
        return (i + 1) % self.every == 0

# ────────────────── Checkpoint-enable helper for HF models ─────────
def enable_grad_ckpt(model):
    try:                       # huggingface style
        model.gradient_checkpointing_enable()
    except AttributeError:     # fall back to .apply() on sub-modules
        import functools, torch.utils.checkpoint as cp
        def ckpt(m):
            if len(list(m.children())) == 0:     # leaf
                return m
            return cp.CheckpointFunction.apply(m)
        model.apply(ckpt)
# -----------------------------------------------------------------------------
# Main training ---------------------------------------------------------------
# -----------------------------------------------------------------------------

def main(a):
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.output_dir, exist_ok=True)

    tr_lo, va_lo, _ = get_combined_loader(a.data_path, a.seed, a.batch_size)
    n_spk = len(AudioDataset.spk2idx)
    base  = Wav2Vec2BERT('facebook/w2v-bert-2.0').to(device)
    if a.use_gc:
        print("✓ Gradient-checkpointing ON")
        enable_grad_ckpt(base.wav2vec2bert)
    model = WAV2V_PRIV(base, n_spk, use_grl=a.use_grl, lamb=a.lambda_schedule[0], null_rank=a.null_rank).to(device)
    fx = AutoFeatureExtractor.from_pretrained('facebook/w2v-bert-2.0'); sr=16000

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    # ------------- TRAIN -----------------------------------------------------
    if a.train:
        # accum = Accumulator(a.grad_accum)
        # ------------------------------------------------------------
        # VICReg weight schedule   (epoch : (sim , var , cov))
        # edit here if you want different numbers
        # ------------------------------------------------------------
        vic_sched = {
            0: (10, 1, 1),
            1: (15, 5, 5),
            2: (20, 5, 5),     
            3: (25, 8, 8),
        }
        for ep in range(a.epochs):
                        # ---------------- VICReg weights for this epoch --------------
            vic_sim_ep, vic_var_ep, vic_cov_ep = vic_sched.get(
                ep,                                   # default = keep previous CLI values
                (a.vic_sim, a.vic_var, a.vic_cov)
            )

            model.train(); model.lamb = a.lambda_schedule[ep]
            ep_loss=0; probs=[]; labels=[]
            opt.zero_grad(set_to_none=True)
            total_steps = len(tr_lo)
            # for wav,y,spk,_ in tqdm(tr_lo, desc=f'ep{ep}'):
            for step, (wav, y, spk, _) in enumerate(tqdm(tr_lo, desc=f"ep{ep}", total=total_steps)):
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
                    loss += vicreg_loss(z1, z2, vic_sim_ep, vic_var_ep, vic_cov_ep)
                    # loss += vicreg_loss(z1, z2, a.vic_sim, a.vic_var, a.vic_cov)

                (loss / a.grad_accum).backward()            ### NEW  (scaled)

                last_batch = (step + 1 == len(tr_lo))       ### NEW
                # if accum.step_now(step, last_batch):        ### NEW
         
                need_step = ((step + 1) % a.grad_accum == 0) or (step + 1 == total_steps)
                if need_step:
                    opt.step()
                    opt.zero_grad(set_to_none=True)

                # opt.zero_grad()
                # loss.backward()
                # opt.step()
                ep_loss += loss.item(); probs.extend(spoof.softmax(1)[:,1].detach().cpu()); labels.extend(y.cpu())
            print(f"ep{ep} loss={ep_loss/len(tr_lo):.4f} AUROC={roc_auc_score(labels, probs):.3f}")
            update_null(model, tr_lo, fx, sr, a.null_rank) if a.null_rank else None
            eval_acc, eval_auroc, eval_eer = run_validation(model.base, fx, va_lo, sr)
            print(f"Epoch {ep}: Final Validation Accuracy: {eval_acc}, Final Validation AUROC: {eval_auroc}, Final Validation EER: {eval_eer}")
            ck = f"{a.dataset_name}_ep{ep}_{datetime.now():%Y%m%d_%H%M%S}.pth"
            torch.save(model.state_dict(), os.path.join(a.output_dir, ck))
            


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
    
    p.add_argument("--grad_accum", type=int, default=1,
               help="Accumulate gradients over N mini-batches")
# --use_gc already exists; we’ll wire it up below
    p.add_argument("--use_vicreg", type=bool, default=False, help="Enable Gradient-checkpoing")
    # ------------------------------------------------------------------
    # VicReg weights  
    # ------------------------------------------------------------------
    p.add_argument("--vic_sim", type=float, default=25.0,
                help="Weight of the similarity term in VICReg loss")
    p.add_argument("--vic_var", type=float, default=1.0,
                help="Weight of the variance term in VICReg loss")
    p.add_argument("--vic_cov", type=float, default=1.0,
                help="Weight of the covariance term in VICReg loss")
    
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
     # --------- parse sigma schedule ---------                              # <<< NEW >>>
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




# nohup python ../apm0046210-synthetic-speech-detection-train-main/ssd_train/train_grl_vib_vic.py --data_path "../data/ASVspoof2019_LA/" "../data/in_the_wild/" "../data/ASVspoof2021_LA_eval/" "../data/wavefake/" "../data/LJSpeech-1.1/" --train True --batch_size 24 --use_vicreg True --epochs 3 --amp_dtype bf16 --dataset_name "combined_hybrid_vic_e3_relu_b72__wt_for-norm">combined_hybrid_vic_e3_relu_b72__wt_for-norm.log &

# nohup python ../apm0046210-synthetic-speech-detection-train-main/ssd_train/train_grl_vib_vic.py --data_path "../data/ASVspoof2019_LA/" "../data/in_the_wild/" "../data/ASVspoof2021_LA_eval/" "../data/wavefake/" "../data/LJSpeech-1.1/" --train True --batch_size 288 --use_vicreg True --epochs 3 --amp_dtype bf16 --grad_accum 4 --use_gc True --dataset_name "combined_hybrid_vic_e3_relu_b72__wt_for-norm_test_temp">combined_hybrid_vic_e3_relu_b72__wt_for-norm_test_temp.log &

# nohup python ../apm0046210-synthetic-speech-detection-train-main/ssd_train/train_grl_vib_vic.py --data_path "../data/ASVspoof2019_LA/" "../data/in_the_wild/" "../data/ASVspoof2021_LA_eval/" "../data/wavefake/" "../data/LJSpeech-1.1/" --train True --batch_size 28 --use_vicreg True --epochs 3 --amp_dtype bf16 --dataset_name "combined_hybrid_vic_schedule_e3_relu_b72__wt_for-norm">combined_hybrid_vic_schedule_e3_relu_b72__wt_for-norm.log &

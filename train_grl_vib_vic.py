# ============================================================
# Wav2Vec2-BERT  •  Spoof vs Real
#   Optional Gradient-Reversal (GRL) speaker adversary
#   Optional Speaker-Null projection
# ============================================================

import argparse, os, random
from datetime import datetime
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Function
from tqdm import tqdm
from transformers import AutoFeatureExtractor
from sklearn.metrics import roc_auc_score

from models import Wav2Vec2BERT
from data_utils_speaker import (
    get_combined_loader, get_test_loader, run_validation, AudioDataset
)

# ---------------- Device ----------------
device = ("cuda" if torch.cuda.is_available() else
          "mps"  if getattr(torch.backends, "mps", None)
                    and torch.backends.mps.is_available() else "cpu")

# ---------------- GRL op ----------------
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lamb): ctx.lamb = lamb; return x
    @staticmethod
    def backward(ctx, g):      return -ctx.lamb * g, None
def grad_reverse(x, lamb=0.5): return GradReverse.apply(x, lamb)

class SpkHead(nn.Module):
    """768 → 512 → 256 → n_spk"""
    def __init__(self, dim, n_spk):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 768), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(768, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_spk)
        )
    def forward(self, x): return self.net(x)


# structured_noise.py
import torch, torch.nn as nn

class StructuredNoise(nn.Module):
    """
    Adds (1) channel dropout  and  (2) per-dim scaled Gaussian noise.
    Only active during .train().
    """
    def __init__(self, p_drop=0.15, sigma=0.02, momentum=0.1, feat_dim=1024):
        super().__init__()
        self.p_drop = p_drop
        self.sigma = sigma
        # running estimate of per-dim stdev for heteroskedastic scaling
        self.register_buffer("running_std", torch.ones(feat_dim))
        self.momentum = momentum

    def forward(self, x: torch.Tensor):
        if not self.training or self.sigma == 0.0 and self.p_drop == 0.0:
            return x

        # -------- 1. Channel drop-mask ----------
        if self.p_drop > 0:
            keep_mask = torch.rand_like(x) > self.p_drop    # Bernoulli(1-p)
            x = x * keep_mask

        # -------- 2. Heteroskedastic Gaussian ----
        if self.sigma > 0:
            # update running std (on current batch)
            batch_std = x.detach().std(dim=0)
            self.running_std = (1 - self.momentum) * self.running_std + \
                               self.momentum * batch_std
            noise = torch.randn_like(x) * self.running_std * self.sigma
            x = x + noise
        return x

# ---------------- Wrapper ---------------
class WAV2V_PRIV(nn.Module):
    """
    * base encoder + spoof classifier (always on)
    * optional GRL speaker adversary     (--use_grl)
    * optional Speaker-Null projection   (--null_rank > 0)
    """
    def __init__(self, base: Wav2Vec2BERT, n_spk: int,
                 use_grl: bool, lamb: float, null_rank: int):
        super().__init__()
        self.base = base
        self.use_grl   = use_grl
        self.lamb      = lamb
        self.null_rank = null_rank
        H = base.config.hidden_size
        # self.noise = StructuredNoise(p_drop=0.15,sigma=0.02,feat_dim=base.config.hidden_size)
        self.noise = StructuredNoise(p_drop=0.15,sigma=0.02,feat_dim=2*H)

        # dim = base.config.hidden_size
        

        if use_grl:
            self.spk_head = SpkHead(2*H, n_spk)
            # self.spk_head = SpkHead(dim, n_spk)

        if null_rank > 0:
            self.register_buffer("U", torch.zeros(dim, null_rank))
            self.U_valid = False  # updated after epoch-0

    # ---------- helper ----------
    def apply_null(self, emb):
        if self.null_rank > 0 and self.U_valid:
            return emb - emb @ self.U @ self.U.T
        return emb

    def forward(self, feats, ret_emb=False):
        out = self.base.wav2vec2bert(
            feats["input_features"],
            attention_mask=feats.get("attention_mask"),
            output_hidden_states=True,
            return_dict=True,
        )

        H_states = out.last_hidden_state
        # ---------- Spoof branch ----------
        mean_emb = self.base.merged_strategy(H_states)      # (B,H)  (mean)
        mean_emb = self.apply_null(mean_emb)
        rf_logits = self.base.classifier(mean_emb)

        # emb = self.base.merged_strategy(out.last_hidden_state)
        # emb = self.apply_null(emb)
        # rf_logits = self.base.classifier(emb)

        if self.use_grl:
            # sigma = 0.02
            # noisy_emb = emb+torch.randn_like(emb)*sigma
            std_emb  = H_states.std(dim=1, unbiased=False)  # (B,H)
            spk_vec  = torch.cat([mean_emb, std_emb], dim=-1)   # (B,2H)
            spk_vec  = self.noise(spk_vec)
            # spk_vec = spk_vec+torch.randn_like(spk_vec)*sigma
            spk_logits = self.spk_head(grad_reverse(spk_vec, self.lamb))

            # noisy_emb = self.noise(emb)

            # spk_logits = self.spk_head(grad_reverse(noisy_emb, self.lamb))
            # spk_logits = self.spk_head(grad_reverse(emb, self.lamb))
        else:
            spk_logits = None

        if ret_emb:
            return rf_logits, spk_logits, emb
        return rf_logits, spk_logits

# ---------------- null-space builder ----------------
@torch.no_grad()
def update_null(model: WAV2V_PRIV, loader, fx, sr, rank):
    feats, spks = [], []
    for wav, y, spk, _ in loader:
        mask = y == 1
        if not mask.any(): continue
        w = fx(wav[mask].to(torch.float32).numpy(), sampling_rate=sr,
               return_attention_mask=True, padding_value=0, return_tensors="pt").to(device)
        h = model.base.wav2vec2bert(w.input_features,
                                    attention_mask=w.attention_mask,
                                    output_hidden_states=True).last_hidden_state
        e = model.base.merged_strategy(h)
        feats.append(e.cpu()); spks.extend(spk[mask].tolist())
    X = torch.cat(feats)                      # N×D
    spks = torch.tensor(spks)
    mu_all = X.mean(0, keepdim=True)
    C = torch.zeros(X.size(1), X.size(1))
    for s in spks.unique():
        xs = X[spks == s]
        mu_s = xs.mean(0, keepdim=True)
        C += (mu_s - mu_all).T @ (mu_s - mu_all)
    _, V = torch.linalg.eigh(C)
    model.U.copy_(V[:, -rank:]); model.U_valid = True

# ---------------- Main ------------------
def main(a):

    print("Arguments:")
    for key, value in vars(a).items(): 
        print(f" {key}: {value}")

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.output_dir, exist_ok=True)

    snr_db=float(a.snr_db)
    pitch_shift=float(a.pitch_shift)
    tempo_rate=float(a.tempo_rate)
    formant_alpha=float(a.formant_alpha)

    lambda_schedule = a.lambda_schedule

    tr_loader, val_loader, _ = get_combined_loader(a.data_path, a.seed, a.batch_size)
    n_spk = len(AudioDataset.spk2idx)
    print("n_speakers =", n_spk)

    base  = Wav2Vec2BERT("facebook/w2v-bert-2.0").to(device)
    model = WAV2V_PRIV(base, n_spk,
                       use_grl=a.use_grl,
                       lamb=lambda_schedule[0],
                       null_rank=a.null_rank).to(device)
    fx  = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    sr  = 16000


    # ---------- AMP set-up ----------
    # ------------ AMP utility -------------
    use_amp   = (a.amp_dtype != "none")
    is_bf16   = (a.amp_dtype == "bf16")
    dtype_amp = torch.bfloat16 if is_bf16 else torch.float16

    if use_amp:
        if device == "cuda":
            autocast = lambda: torch.cuda.amp.autocast(dtype=dtype_amp)
            scaler   = torch.amp.GradScaler(enabled=not is_bf16)
        else:  # mps / cpu
            autocast = lambda: torch.autocast(device_type=device, dtype=dtype_amp)
            scaler   = torch.amp.GradScaler(enabled=False)
    else:
        from contextlib import nullcontext
        autocast = nullcontext
        scaler   = torch.amp.GradScaler(enabled=False)          # placeholder

    # ---- two LR groups only if GRL is on ----
    param_groups = [{"params": [p for n,p in model.named_parameters()
                                 if not n.startswith("spk_head.")],
                     "lr": a.lr}]
    if a.use_grl:
        param_groups.append({"params": model.spk_head.parameters(),
                             "lr": a.lr_spk})
    opt = torch.optim.AdamW(param_groups, weight_decay=a.weight_decay)

    if a.checkpoint:                                          # << NEW >>
        ckpt_path = (a.checkpoint if os.path.isfile(a.checkpoint)
                    else os.path.join(a.output_dir, a.checkpoint))
        print(f"✔ Loading checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        def expand_weight(w_old, factor=2):
            """Duplicate along input dimension and divide by factor so variance ≈ constant."""
            return torch.cat([w_old] * factor, dim=1) / factor


        key = "spk_head.net.0.weight"
        if key in state_dict and state_dict[key].shape[1] * 2 == model.spk_head.net[0].weight.shape[1]:
            state_dict[key] = expand_weight(state_dict[key])          # 768×H → 768×2H
            print("✓ expanded speaker weight")

        bias_key = "spk_head.net.0.bias"                  # unchanged size; keep as is

        # Remove this one if you are not adding structured noise
        if "noise.running_std" not in state_dict:
            state_dict["noise.running_std"] = torch.ones(model.noise.running_std.shape)
        model.load_state_dict(state_dict)


        # ========== TRAIN ==========
    if a.train:
        model.train()

        schedule = {
            0:(0.5,0.00,0.000),
            1:(0.75,0.00,0.000),
            2:(1.0,0.00,0.005),
            3:(1.25,0.00,0.005),
            4:(1.5,0.00,0.010),
            5:(2.0,0.00,0.010),
            6:(2.0,0.00,0.020),
            7:(2.0,0.00,0.020),
            8:(2.0,0.00,0.020),
            9:(2.0,0.00,0.020),
        }
        for ep in range(a.epochs):
                                            
            model.lamb = schedule[ep][0]
            model.noise.p_drop=schedule[ep][1]
            model.noise.sigma=schedule[ep][2]

            # model.lamb = lambda_schedule[ep]
            # if ep==0:
            #     model.noise.p_drop=0.05
            #     model.noise.sigma=0.005
            # elif ep==1:
            #     model.noise.p_drop=0.10
            #     model.noise.sigma=0.01 
            # else:
            #     model.noise.p_drop=0.15
            #     model.noise.sigma=0.02

            print(f"\nEpoch {ep} — λ_adv = {model.lamb}")
            print(f"\nEpoch {ep} — noise p_drop = {model.noise.p_drop}")
            print(f"\nEpoch {ep} — noise sigma = {model.noise.sigma}")
            running, pr, gt = 0.0, [], []
            for step,(wav,y,spk,_) in enumerate(tqdm(tr_loader, desc=f"ep{ep}")):
                feats = fx(wav.float().numpy(), sampling_rate=sr,
                           return_attention_mask=True, padding_value=0,
                           return_tensors="pt")
                feats = {k:v.to(device) for k,v in feats.items()}
                y, spk = y.to(device), spk.to(device)

                # ---- forward with / without autocast ----
                if use_amp:
                    with autocast():
                        rf_logits, spk_logits = model(feats)[:2]
                        loss = F.cross_entropy(rf_logits, y)
                        if a.use_grl:
                            m = y == 1
                            if m.any():
                                loss += F.cross_entropy(spk_logits[m], spk[m])
                    if device == "cuda" and not is_bf16:
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    else:
                        loss.backward()
                        opt.step()
                    opt.zero_grad(set_to_none=True)
                else:
                    rf_logits, spk_logits, _ = model(feats)
                    loss = F.cross_entropy(rf_logits, y)
                    if a.use_grl:
                        m = y == 1
                        if m.any():
                            loss += F.cross_entropy(spk_logits[m], spk[m])
                    loss.backward(); opt.step(); opt.zero_grad()

                running += loss.item()
                pr.extend(rf_logits.softmax(-1)[:,1].detach().cpu().tolist())
                gt.extend(y.cpu().tolist())

                if (step+1) % a.eval_steps == 0:
                    run_validation(model.base, fx, val_loader, sr); model.train()

            print(f"ep{ep}  loss={running/(step+1):.4f}  AUROC={roc_auc_score(gt,pr):.4f}")


                    # build/update null space
            if a.null_rank > 0:
                update_null(model, tr_loader, fx, sr, a.null_rank)
                print(f"  ↪ updated speaker-null (rank {a.null_rank})")
            
            eval_acc, eval_auroc, eval_eer = run_validation(model.base, fx, val_loader, sr)
            print(f"Epoch {ep}: Final Validation Accuracy: {eval_acc}, Final Validation AUROC: {eval_auroc}, Final Validation EER: {eval_eer}")
            ck = f"{a.dataset_name}_ep{ep}_{datetime.now():%Y%m%d_%H%M%S}.pth"
            torch.save(model.state_dict(), os.path.join(a.output_dir, ck))



    # ------------- testing --------------
    if a.test:
        if not a.checkpoint: raise ValueError("--checkpoint required for --test")
        test_loader = get_test_loader(a.data_path, a.seed, a.batch_size)
        model.load_state_dict(torch.load(os.path.join(a.output_dir, a.checkpoint),
                                         map_location=device), strict=False)
        model.eval(); run_validation(model.base, fx, test_loader, sr)

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


# nohup python ../apm0046210-synthetic-speech-detection-train-main/ssd_train/finetune_with_ckpt_grl.py --data_path "../data/ASVspoof2019_LA/" "../data/in_the_wild/" "../data/ASVspoof2021_LA_eval/" "../data/wavefake/" "../data/LJSpeech-1.1/" --train True --batch_size 72 --epochs 10 --use_grl True --amp_dtype bf16 --dataset_name "combined_hybrid_lambda_schedule0.5_2.0_speaker_4layer_e10_relu_b72_wt_for-norm_grl_mean_plus_std_with_noise_schedule0.00-0.02" >combined_hybrid_lambda_schedule0.5_2.0_speaker_4layer_e10_relu_b72_wt_for-norm_grl_mean_plus_std_with_noise_schedule0.00_0.02.log &

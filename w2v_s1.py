# ============================================================
# Wav2Vec2-BERT  •  Spoof vs Real
#   Optional Gradient-Reversal (GRL) speaker adversary
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
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x
    @staticmethod
    def backward(ctx, g):
        return -ctx.lamb * g, None
def grad_reverse(x, lamb=0.5):
    return GradReverse.apply(x, lamb)

# ---------------- Speaker MLP ----------—

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

# ---------------- Wrapper ---------------
class WAV2V_PRIV(nn.Module):
    """
    * base encoder + spoof classifier (always on)
    * optional GRL speaker adversary     (--use_grl)
    """
    def __init__(self, base: Wav2Vec2BERT, n_spk: int,
                 use_grl: bool, lamb: float):
        super().__init__()
        self.base = base
        self.use_grl   = use_grl
        self.lamb      = lamb

        dim = base.config.hidden_size
        if use_grl:
            self.spk_head = SpkHead(dim, n_spk)

    def forward(self, feats, ret_emb=False):
        out = self.base.wav2vec2bert(
            feats["input_features"],
            attention_mask=feats.get("attention_mask"),
            output_hidden_states=True,
            return_dict=True,
        )
        emb = self.base.merged_strategy(out.last_hidden_state)

        synth_logits = self.base.classifier(emb)

        if self.use_grl:
            spk_logits = self.spk_head(grad_reverse(emb, self.lamb))
        else:
            spk_logits = None

        return synth_logits, spk_logits

# ---------------- Main ------------------
def main(a):
    print("Arguments:")
    for key, value in vars(a).items():
        print(f" {key}: {value}")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.output_dir, exist_ok=True)

    lambda_schedule = a.lambda_schedule

    tr_loader, val_loader, _ = get_combined_loader(a.data_path, a.seed, a.batch_size)
    n_spk = len(AudioDataset.spk2idx)
    print("n_speakers =", n_spk)

    base  = Wav2Vec2BERT("facebook/w2v-bert-2.0").to(device)
    model = WAV2V_PRIV(base, n_spk,
                       use_grl=a.use_grl,
                       lamb=lambda_schedule[0]).to(device)
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



        # ========== TRAIN ==========
    if a.train:
        model.train()
        for ep in range(a.epochs):
            model.lamb = lambda_schedule[ep]                                   
            print(f"\nEpoch {ep} — λ_adv = {model.lamb}")
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
            
            eval_acc, eval_auroc, eval_eer = run_validation(model.base, fx, val_loader, sr)
            print(f"Epoch {ep}: Final Validation Accuracy: {eval_acc}, Final Validation AUROC: {eval_auroc}, Final Validation EER: {eval_eer}")
            if ep==a.epochs:
                ck = f"{a.dataset_name}_ep{ep}_{datetime.now():%Y%m%d_%H%M}.pth"
                torch.save(model.state_dict(), os.path.join(a.output_dir, ck))


    # ------------- testing --------------
    if a.test:
        if not a.checkpoint: raise ValueError("--checkpoint required for --test")
        test_loader = get_test_loader(a.data_path, a.seed, a.batch_size)
        model.load_state_dict(torch.load(os.path.join(a.output_dir, a.checkpoint),
                                         map_location=device))
        model.eval(); run_validation(model.base, fx, test_loader, sr)

# ---------------- CLI -------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser("W2v-BERT  +  GRL toggles")
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

    p.add_argument("--amp_dtype", choices=["none","fp16","bf16"], default="none",
                   help="Automatic Mixed Precision type")

    args = p.parse_args()

    # --------- parse λ schedule ---------
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

    args.lambda_schedule = lam_list                                           
    main(args)

    # nohup python ../apm0046210-synthetic-speech-detection-train-main/ssd_train/train_grl_w2v2bert.py --data_path "../data/ASVspoof2019_LA/" "../data/in_the_wild/" "../data/ASVspoof2021_LA_eval/" "../data/wavefake/" "../data/LJSpeech-1.1/" --train True --batch_size 72 --epochs 5 --lambda_adv 2 --use_grl True --amp_dtype bf16 --dataset_name "combined_hybrid_lambda_schdule_deep_speaker_network_4layer_e5_relu_b72__wt_for-norm_test" >combined_hybrid_lambda_schdule_deep_speaker_network_4layer_e5_relu_b72_wt_for-norm_test.log &
    # nohup python ../apm0046210-synthetic-speech-detection-train-main/ssd_train/train_grl_w2v2bert.py --data_path "../data/ASVspoof2019_LA/" "../data/in_the_wild/" "../data/ASVspoof2021_LA_eval/" "../data/wavefake/" "../data/LJSpeech-1.1/" --train True --batch_size 72 --epochs 5 --lambda_adv 0.5,0.75,1,1.5,2 --use_grl True --amp_dtype bf16 --dataset_name "combined_hybrid_lambda_schdule_deep_speaker_network_4layer_e5_relu_b72__wt_for-norm_08_04" >combined_hybrid_lambda_schdule_deep_speaker_network_4layer_e5_relu_b72_wt_for-norm_08_04.log &
    # nohup python ../apm0046210-synthetic-speech-detection-train-main/ssd_train/train_grl_w2v2bert.py --data_path "../data/ASVspoof2019_LA/" "../data/in_the_wild/" "../data/ASVspoof2021_LA_eval/" "../data/wavefake/" "../data/LJSpeech-1.1/" --train True --batch_size 72 --epochs 5 --lambda_adv 0.5,0.75,1,1.5,2 --use_grl True --amp_dtype bf16 --dataset_name "grl_stage1_lambda_schedule_speaker_network_4layer_e5_relu_b72" >grl_stage1_lambda_schedule_speaker_network_4layer_e5_relu_b72.log &

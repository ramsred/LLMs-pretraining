# ------------------------------------------------------------
# Train Wav2Vec2BERT with Gradient-Reversal Speaker Adversary
# ------------------------------------------------------------
#  • Keeps original `Wav2Vec2BERT` and `ClassificationHead` untouched
#  • Adds a tiny wrapper that plugs a Gradient-Reversal Layer (GRL)
#    and a speaker-ID head
#  • Expects AudioDataset to return (wave, label, speaker_idx, key)
# ------------------------------------------------------------

import argparse, os, random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from tqdm import tqdm
from transformers import AutoFeatureExtractor
from sklearn.metrics import roc_auc_score

from models import Wav2Vec2BERT            # original model file
from data_utils_speaker import (          # speaker-aware utils from canvas
    get_combined_loader,
    get_test_loader,
    run_validation,
    AudioDataset,
)
# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

device = (
    "cuda" if torch.cuda.is_available() else (
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)

# ------------------------------------------------------------
# Gradient Reversal Layer
# ------------------------------------------------------------
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return -ctx.lamb * g, None

def grad_reverse(x, lamb=0.5):
    return GradReverse.apply(x, lamb)

# ------------------------------------------------------------
# Wrapper network with speaker head
# ------------------------------------------------------------
class Wav2Vec2BERT_GRL(nn.Module):
    def __init__(self, base: Wav2Vec2BERT, n_speakers: int, grl_lambda: float = 0.5):
        super().__init__()
        self.base = base                # frozen or fine-tuned encoder+task head
        self.grl_lambda = grl_lambda
        emb_dim = base.config.hidden_size
        self.spk_head = nn.Linear(emb_dim, n_speakers)

    def forward(self, inputs, speaker=True):
        # base forward needs hidden states
        out = self.base.wav2vec2bert(
            inputs["input_features"],
            attention_mask=inputs.get("attention_mask"),
            output_hidden_states=True,
            return_dict=True,
        )
        h = self.base.merged_strategy(out.last_hidden_state)  # [B,768]
        rf_logits = self.base.classifier(h)

        if speaker:
            h_rev = grad_reverse(h, self.grl_lambda)
            spk_logits = self.spk_head(h_rev)
            return rf_logits, spk_logits
        else:
            return rf_logits

# ------------------------------------------------------------
# Main training / testing
# ------------------------------------------------------------

def main(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ----------------- model + feature extractor -------------
    base_model = Wav2Vec2BERT("facebook/w2v-bert-2.0").to(device)
    # build loaders first to know n_speakers
    tr_loader, val_loader, _ = get_combined_loader(args.data_path, args.seed, args.batch_size)
    if not hasattr(AudioDataset, 'spk2idx'):
        raise AttributeError("AudioDataset must define 'spk2idx' as a class-level attribute.")
    n_speakers = len(AudioDataset.spk2idx)
    model = Wav2Vec2BERT_GRL(base_model, n_speakers, grl_lambda=args.lambda_adv).to(device)
    fx = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    sr = 16000

    # ----------------- optimizer -----------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ----------------- training ------------------------------
    if args.train:
        model.train()
        for ep in range(args.epochs):
            probs, gts = [], []
            running = 0.0
            for step, (wav, lbl, spk, _) in enumerate(tqdm(tr_loader, desc=f"ep{ep}")):
                wav = wav.to(torch.float32)
                inp = fx(wav.numpy(), sampling_rate=sr, return_attention_mask=True,
                         padding_value=0, return_tensors="pt")
                inp = {k: v.to(device) for k,v in inp.items()}
                # inp["input_features"] = inp.pop("input_values")  # rename as model expects

                lbl = lbl.to(device); spk = spk.to(device)

                rf_logits, spk_logits = model(inp, speaker=True)

                loss_rf  = F.cross_entropy(rf_logits, lbl)
                loss_spk = F.cross_entropy(spk_logits, spk)
                loss = loss_rf + loss_spk   # GRL flips sign internally

                optimizer.zero_grad(); loss.backward(); optimizer.step()
                running += loss.item()

                probs.extend(rf_logits.softmax(-1)[:,1].detach().cpu().tolist())
                gts.extend(lbl.tolist())

                if (step+1)%args.eval_steps==0:
                    run_validation(model.base, fx, val_loader, sr)  # evaluate task head only
                    model.train()
            auroc = roc_auc_score(gts, probs)
            print(f"ep{ep}  loss={running/(step+1):.4f}  AUROC={auroc:.4f}")
            ck = f"{args.dataset_name}_grl_ep{ep}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            torch.save(model.state_dict(), os.path.join(args.output_dir, ck))

    # ----------------- testing --------------------------------
    if args.test:
        if not args.checkpoint:
            raise ValueError("Checkpoint file must be specified for testing.")
        test_loader = get_test_loader(args.data_path, args.seed, args.batch_size)
        model.load_state_dict(torch.load(os.path.join(args.output_dir, args.checkpoint), map_location=device))
        model.eval()
        run_validation(model.base, fx, test_loader, sr)

# ------------------------------------------------------------
# CLI ---------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser("Train W2v-BERT with GRL speaker adversary")
    ap.add_argument("--data_path", nargs="+", required=True)
    ap.add_argument("--output_dir", default="./ckpt")
    ap.add_argument("--train", type=bool, default=False, help="Flag to run training") 
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--weight_decay", type=float, default=5e-5)
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--dataset_name", default="dataset")
    ap.add_argument("--lambda_adv", type=float, default=0.5, help="GRL lambda")
    args = ap.parse_args(); main(args)
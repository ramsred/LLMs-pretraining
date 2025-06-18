# ------------------------------------------------------------
# Train Wav2Vec2BERT + GRL speaker adversary
#   * label: 1 = bonafide / human / real
#   * label: 0 = spoof / synthetic / fake
#   * GRL speaker-loss computed ONLY on label == 1
# ------------------------------------------------------------
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
device = "cuda" if torch.cuda.is_available() else (
         "mps"  if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
         else "cpu")

# ---------------- GRL -------------------
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lamb): ctx.lamb = lamb; return x.view_as(x)
    @staticmethod
    def backward(ctx, g):       return -ctx.lamb * g, None
def grad_reverse(x, lamb=0.5): return GradReverse.apply(x, lamb)

# ---------------- Wrapper ----------------
class Wav2Vec2BERT_GRL(nn.Module):
    def __init__(self, base: Wav2Vec2BERT, n_spk: int, lamb: float = .5):
        super().__init__(); self.base = base; self.lamb = lamb
        self.spk_head = nn.Linear(base.config.hidden_size, n_spk)

    def forward(self, feats):
        out = self.base.wav2vec2bert(
            feats["input_features"],
            attention_mask=feats.get("attention_mask"),
            output_hidden_states=True,
            return_dict=True,
        )
        emb = self.base.merged_strategy(out.last_hidden_state)
        rf_logits = self.base.classifier(emb)
        spk_logits = self.spk_head(grad_reverse(emb, self.lamb))
        return rf_logits, spk_logits

# ---------------- Main ------------------
def main(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.output_dir, exist_ok=True)

    tr_loader, val_loader, _ = get_combined_loader(a.data_path, a.seed, a.batch_size)
    n_spk = len(AudioDataset.spk2idx)

    base = Wav2Vec2BERT("facebook/w2v-bert-2.0").to(device)
    model = Wav2Vec2BERT_GRL(base, n_spk, lamb=a.lambda_adv).to(device)
    fx = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0"); sr = 16000
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    # ------------- training -------------
    if a.train:
        model.train()
        for ep in range(a.epochs):
            running, pr, gt = 0.0, [], []
            for step,(wav,y,spk,_) in enumerate(tqdm(tr_loader,desc=f"ep{ep}")):
                assert y.max()<=1 and y.min()>=0, "Labels must be 0 (spoof) or 1 (bonafide)"
                feats = fx(wav.to(torch.float32).numpy(), sampling_rate=sr,
                           return_attention_mask=True, padding_value=0,
                           return_tensors="pt")
                feats = {k:v.to(device) for k,v in feats.items()}
                y, spk = y.to(device), spk.to(device)

                rf_logits, spk_logits = model(feats)
                loss_rf  = F.cross_entropy(rf_logits, y)

                # ---- speaker loss only for real clips ----
                mask = y == 1
                loss_spk = F.cross_entropy(spk_logits[mask], spk[mask]) if mask.any() \
                           else torch.tensor(0., device=device)

                loss = loss_rf + loss_spk
                opt.zero_grad(); loss.backward(); opt.step()
                running += loss.item()

                pr.extend(rf_logits.softmax(-1)[:,1].detach().cpu().tolist())
                gt.extend(y.cpu().tolist())

                if (step+1) % a.eval_steps == 0:
                    run_validation(model.base, fx, val_loader, sr); model.train()

            print(f"ep{ep}  loss={running/(step+1):.4f}  AUROC={roc_auc_score(gt,pr):.4f}")
            eval_acc, eval_auroc, eval_eer = run_validation(model.base, fx, val_loader, sr)
            ck = f"{a.dataset_name}_grl_ep{ep}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            torch.save(model.state_dict(), os.path.join(a.output_dir, ck))

    # ------------- testing --------------
    if a.test:
        if not a.checkpoint: raise ValueError("--checkpoint required when --test")
        test_loader = get_test_loader(a.data_path, a.seed, a.batch_size)
        model.load_state_dict(torch.load(os.path.join(a.output_dir, a.checkpoint),
                                         map_location=device), strict=False)
        model.eval(); run_validation(model.base, fx, test_loader, sr)



# ---------------- CLI -------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser("W2v-BERT with GRL (real speakers only)")
    p.add_argument("--data_path", nargs="+", required=True)
    p.add_argument("--output_dir", default="./ckpt")
    p.add_argument("--train", type=bool, default=False, help="Flag to run training")
    p.add_argument("--test", type=bool, default=False, help="Flag to run training")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=5e-5)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dataset_name", default="dataset")
    p.add_argument("--lambda_adv", type=float, default=0.5)
    args = p.parse_args(); main(args)


import argparse
import sys
import os
import numpy as np
from tqdm import tqdm
import random
import librosa  # noqa: F401  – required by data_utils
import torch
from transformers import AutoFeatureExtractor
from sklearn.metrics import roc_auc_score
from collections import defaultdict, Counter  # noqa: F401 – kept from original for future use
from models import Wav2Vec2BERT
from data_utils import (
    get_combined_loader,
    get_test_loader,
    pad,
    seed_worker,
    AudioDataset,
    compute_det_curve,
    compute_eer,
    run_validation,
)
from datetime import datetime
from opacus import PrivacyEngine

# ------------------------------------------------------------
# Device setup
# ------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "mps"

# ------------------------------------------------------------
# Main training / evaluation script
# ------------------------------------------------------------

def main(args):
    # -------------------- Reproducibility --------------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # -------------------- Output directories -----------------
    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------- Model init -------------------------
    if args.pretrained_model_name.lower() == "wave2vec2bert":
        model_name = "facebook/w2v-bert-2.0"
        model = Wav2Vec2BERT(model_name)
        sampling_rate = 16000
    else:
        raise ValueError(f"Model {args.pretrained_model_name} not supported")

    model = model.to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    # -------------------- Training ---------------------------
    if args.train:
        print(f"[INFO] Starting fine‑tuning | device={device}")
        trn_loader, val_loader, _ = get_combined_loader(
            args.data_path, args.seed, args.batch_size
        )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        # ------------- Optional: Differential Privacy ----------
        if args.dp:
            privacy_engine = PrivacyEngine()
            if args.target_epsilon is not None:
                model, optimizer, trn_loader = privacy_engine.make_private_with_epsilon(
                    module=model,
                    optimizer=optimizer,
                    data_loader=trn_loader,
                    target_epsilon=args.target_epsilon,
                    target_delta=args.delta,
                    epochs=args.epochs,
                    max_grad_norm=args.max_grad_norm,
                )
                print(
                    f"[DP] make_private_with_epsilon | ε={args.target_epsilon} | δ={args.delta} | σ={privacy_engine.noise_multiplier:.3f}"
                )
            else:
                print("make private with noise multiplier")
                model, optimizer, trn_loader = privacy_engine.make_private(
                    module=model,
                    optimizer=optimizer,
                    data_loader=trn_loader,
                    noise_multiplier=args.noise_multiplier,
                    max_grad_norm=args.max_grad_norm,
                )
                print(
                    f"[DP] make_private | σ={args.noise_multiplier} | max_grad_norm={args.max_grad_norm}"
                )
        else:
            privacy_engine = None  # not used

        model.train()
        for epoch in range(args.epochs):
            print("=" * 60)
            print(
                f"Epoch {epoch+1}/{args.epochs}  |  lr={args.lr}  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            outputs_list, labels_list = [], []
            running_loss = 0.0

            for step, (batch_x, batch_y, _) in enumerate(
                tqdm(trn_loader, desc="Fine‑tuning")
            ):
                batch_x = batch_x.numpy()
                inputs = feature_extractor(
                    batch_x,
                    sampling_rate=sampling_rate,
                    return_attention_mask=True,
                    padding_value=0,
                    return_tensors="pt",
                ).to(device)
                batch_y = batch_y.to(device)
                inputs["labels"] = batch_y

                outputs = model(**inputs)
                loss = outputs.loss
                running_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # store metrics for this mini‑batch
                probs = outputs.logits.softmax(dim=-1)[:, 1]
                outputs_list.extend(probs.detach().cpu().tolist())
                labels_list.extend(batch_y.detach().cpu().tolist())

                # periodic validation
                if (step + 1) % args.eval_steps == 0:
                    eval_acc, eval_auc, eval_eer = run_validation(
                        model, feature_extractor, val_loader, sr=sampling_rate
                    )
                    print(
                        f"\n[VAL] step {step+1}  |  AUROC={eval_auc:.4f}  |  Acc={eval_acc:.4f}  |  EER={eval_eer[0]:.4f}"
                    )
                    model.train()

            # ----- end epoch -----
            epoch_auc = roc_auc_score(labels_list, outputs_list)
            epoch_eer = compute_eer(np.array(labels_list), np.array(outputs_list))
            print(
                f"[TRAIN] epoch {epoch} done | Loss={running_loss/len(trn_loader):.4f} | AUROC={epoch_auc:.4f} | EER={epoch_eer:.4f}"
            )

            ckpt_name = (
                f"{args.dataset_name}_{args.pretrained_model_name}_epoch_{epoch}_"
                f"{args.lr}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"
            )
            torch.save(model.state_dict(), os.path.join(args.output_dir, ckpt_name))

            # report privacy budget per epoch
            if args.dp:
                eps = privacy_engine.get_epsilon(args.delta)
                print(f"[DP] ε after epoch {epoch}: {eps:.2f} (δ={args.delta})")

        print("[INFO] Fine‑tuning complete.")
        return

    # -------------------- Testing ----------------------------
    if args.test:
        test_loader = get_test_loader(args.data_path[0], args.seed, args.batch_size)
        model_path = os.path.join(args.output_dir, args.checkpoint)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        run_validation(model, feature_extractor, test_loader, sr=sampling_rate)
        print("[INFO] Evaluation complete.")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Wav2Vec2BERT with optional Differential Privacy via Opacus")

    # Basic
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_dir", type=str, default="./ckpt")
    parser.add_argument("--pretrained_model_name", type=str, default="wave2vec2bert")
    parser.add_argument("--data_path", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--dataset_name", type=str, default="dataset")

    # Mode flags
    parser.add_argument("--train", type=bool, default=False, help="Flag to run training")
    parser.add_argument("--eval", type=bool, default=False, help="Flag to run training")
    parser.add_argument("--test", type=bool, default=False, help="Flag to run training")
    parser.add_argument("--checkpoint", type=str, default="")

    # Differential Privacy flags
    parser.add_argument("--dp", action="store_true", help="Enable Opacus DP training")
    parser.add_argument("--noise_multiplier", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--target_epsilon", type=float, default=None)
    parser.add_argument("--delta", type=float, default=1e-5)

    args = parser.parse_args()
    main(args)

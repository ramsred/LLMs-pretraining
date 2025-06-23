import argparse
import sys
import os
import numpy as np
from tqdm import tqdm
import random
import torch
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor
from sklearn.metrics import roc_auc_score
from models import Wav2Vec2BERT
from data_utils import get_combined_loader, get_test_loader, seed_worker, compute_eer, run_validation,laplace_mechanism
from datetime import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"

def main(args):

    print("--data_path : ",args.data_path)
    print("--epochs : ",args.epochs)
    print("--batch_size : ",args.batch_size)
    print("--train : ",args.train)
    print("--test : ",args.test)
    print("--is_hybrid_loss : ",args.is_hybrid_loss)
    print("--use_triplet : ",args.use_triplet)
    print("--margins : ",args.margins)
    print("--triplet_loss_p2 : ",args.triplet_loss_p2)
    print("--triplet_loss_cosine : ",args.triplet_loss_cosine)
    print("--use_mnr_loss_for_hybrid : ",args.use_mnr_loss_for_hybrid)
    print("--num_negatives : ",args.num_negatives)
    print("--epsilon : ",args.epsilon)
    print("--snr_db : ",args.snr_db)
    print("--pitch_shift : ",args.pitch_shift)
    print("--tempo_rate : ",args.tempo_rate)
    print("--formant_alpha : ",args.formant_alpha)

    # Seed for reproducibility
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # Output directories
    output_dir = args.output_dir
    eval_ckpt = os.path.join(output_dir, "eval")

    pretrained_model_name = args.pretrained_model_name
    if pretrained_model_name == 'wave2vec2bert':
        model_name = "facebook/w2v-bert-2.0"
        model = Wav2Vec2BERT(model_name)
        sampling_rate = 16000
    else:
        raise ValueError(f"Model {pretrained_model_name} not supported")
        sys.exit(0)
    
    checkpoint_path = args.checkpoint
    # Load the checkpoint as a fine-tuned model
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model = model.to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    if args.train:
        
        if args.freeze_bert:
            # Model parameters
            for name, param in model.named_parameters():
                # Freeze all layers except the classifier
                if "classifier" not in name:  # Assume classifier layers contain "classifier" in their name
                    print("Freezing Bert layers enabled")
                    param.requires_grad = False
                else:
                    print(f"Unfrozen layer: {name}")

            # Verify which layers are frozen/unfrozen
            for name, param in model.named_parameters():
                print(f"{name}: requires_grad={param.requires_grad}")
            # Print detailed parameter information
            # for name, param in model.named_parameters():
            #     print(f"{name}: {param.numel()} parameters, trainable={param.requires_grad}")

        print(f"Start Finetuning using {device}")

        torch.cuda.empty_cache()
        database_path_list = args.data_path
        print(database_path_list)
        trn_loader, val_loader, test_loader = get_combined_loader(database_path_list, seed, args.batch_size, use_triplet=args.use_triplet,use_mnr=args.use_mnr_loss_for_hybrid,snr_db=args.snr_db, pitch_shift=args.pitch_shift, tempo_rate=args.tempo_rate, formant_alpha=args.formant_alpha)
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model.train()
        # Define alpha progression
        alpha_values = args.dynamic_alpha_list
        if len(alpha_values)!=args.epochs:
            alpha_values = [args.dynamic_alpha_list[0] for alpha in range(args.epochs)]

        # make sure dynamic alpha matches with no of epochs
        # Make sure to print the respective loss triplet or ce  
        current_alpha_index = 0
        for margin in args.margins:
            print(f"Training with margin: {margin}")
            for epoch in range(args.epochs):

                print(f"Epoch {epoch}: Using alpha {alpha_values[current_alpha_index]}")
                # print(f'{output_dir}/{args.dataset_name}_{pretrained_model_name}_epoch_{epoch}_{args.lr}_{datetime.now()}.pth')
                outputs_list, labels_list, train_loss, auroc_list, acc_list, err_list = [], [], [], [], [], []
                num_total, steps = 0, 0
                triplet_loss_list = []
                # Initialize variables for tracking training loss
                total_train_loss = 0.0  # Cumulative loss
                num_batches = 0         # Total number of batches

                for batch in tqdm(trn_loader, desc="Finetuning"):
                    if args.use_triplet:
                        anchor, positive, negative, batch_y = batch
                        # print(f"Anchor shape: {anchor.shape}")
                        # print(f"Positive shape: {positive.shape}")
                        # print(f"Negative shape: {negative.shape}")
                        anchor = anchor.to(device)
                        positive = positive.to(device)
                        negative = negative.to(device)

                        if args.epsilon>0:
                            # print("Laplace Mechanism applied :",args.epsilon)
                            # Apply Laplace Mechanism to triplet features
                            sensitivity_anchor = np.max(anchor.cpu().numpy()) - np.min(anchor.cpu().numpy())
                            sensitivity_positive = np.max(positive.cpu().numpy()) - np.min(positive.cpu().numpy())
                            sensitivity_negative = np.max(negative.cpu().numpy()) - np.min(negative.cpu().numpy())
                            epsilon = args.epsilon  # Privacy budget

                            noisy_anchor = laplace_mechanism(anchor.cpu().numpy(), sensitivity_anchor, epsilon,random_seed=args.seed)
                            noisy_positive = laplace_mechanism(positive.cpu().numpy(), sensitivity_positive, epsilon,random_seed=args.seed)
                            noisy_negative = laplace_mechanism(negative.cpu().numpy(), sensitivity_negative, epsilon,random_seed=args.seed)
                            
                            anchor_inputs = feature_extractor(noisy_anchor, sampling_rate=sampling_rate, return_tensors="pt")['input_features'].to(device)
                            positive_inputs = feature_extractor(noisy_positive, sampling_rate=sampling_rate, return_tensors="pt")['input_features'].to(device)
                            negative_inputs = feature_extractor(noisy_negative, sampling_rate=sampling_rate, return_tensors="pt")['input_features'].to(device)
                        else:
                            # Extract features for anchor, positive, and negative samples
                            anchor_inputs = feature_extractor(anchor.cpu().numpy(), sampling_rate=sampling_rate, return_tensors="pt")['input_features'].to(device)
                            positive_inputs = feature_extractor(positive.cpu().numpy(), sampling_rate=sampling_rate, return_tensors="pt")['input_features'].to(device)
                            negative_inputs = feature_extractor(negative.cpu().numpy(), sampling_rate=sampling_rate, return_tensors="pt")['input_features'].to(device)
                    else:
                        batch_x, batch_y, _ = batch
                        batch_x = batch_x.numpy()
                        if args.epsilon>0:
                            # print("Laplace Mechanism applied :",args.epsilon)
                            # Apply Laplace Mechanism to batch features
                            sensitivity = np.max(batch_x) - np.min(batch_x)
                            epsilon = args.epsilon  # Privacy budget
                            noisy_batch_x = laplace_mechanism(batch_x, sensitivity, epsilon,random_seed=args.seed)

                            inputs = feature_extractor(noisy_batch_x, sampling_rate=sampling_rate, return_attention_mask=True, padding_value=0, return_tensors="pt").to(device)
                        else:

                            inputs = feature_extractor(batch_x, sampling_rate=sampling_rate, return_attention_mask=True, padding_value=0, return_tensors="pt").to(device)
                        batch_y = batch_y.to(device)
                        inputs['labels'] = batch_y

                    if args.use_triplet:
                        outputs = model(anchor_features=anchor_inputs, positive_features=positive_inputs, negative_features=negative_inputs,labels=batch_y, margin=margin,is_hybrid_loss=args.is_hybrid_loss,triplet_loss_p2=args.triplet_loss_p2,triplet_loss_cosine=args.triplet_loss_cosine,use_mnr_loss_for_hybrid=args.use_mnr_loss_for_hybrid,alpha=alpha_values[current_alpha_index])
                    else:
                        outputs = model(**inputs)

                    # Log the triplet loss if applicable
                    # if args.use_triplet and hasattr(outputs, 'triplet_loss'):
                    #     triplet_loss = outputs.triplet_loss.item()
                    #     triplet_loss_list.append(triplet_loss)
                    #     print(f"Triplet Loss: {triplet_loss:.4f}")

                    # Calculate loss
                    total_train_loss += outputs.loss.item()  # Accumulate total loss
                    num_batches += 1  # Increment batch count

                    train_loss.append(outputs.loss.item())

                    if not args.use_triplet:
                        batch_probs = outputs.logits.softmax(dim=-1)
                        outputs_list.extend(batch_probs[:, 1].cpu().tolist())
                        labels_list.extend(batch_y.detach().cpu().numpy().tolist())

                    optim.zero_grad()
                    outputs.loss.backward()
                    optim.step()
                    steps += 1  # Increment steps

                    if steps % args.eval_steps == 0:
                        eval_acc, eval_auroc, eval_eer = run_validation(model, feature_extractor, val_loader, sr=sampling_rate,use_triplet=args.use_triplet,use_mnr=args.use_mnr_loss_for_hybrid,num_negatives=args.num_negatives)
                        auroc_list.append(eval_auroc)
                        acc_list.append(eval_acc)
                        err_list.append(eval_eer[0])
                        model.train()

                if not args.use_triplet:
                    auroc = roc_auc_score(labels_list, outputs_list)
                    eer = compute_eer(np.array(labels_list), np.array(outputs_list))
                    print(f'Training epoch: {epoch} \t AUROC: {auroc} \t EER: {eer}')
                
                # Compute average training loss for the epoch
                avg_train_loss = total_train_loss / num_batches
                print(f"Epoch {epoch}: Average Training Loss: {avg_train_loss:.4f}")

                # Perform final validation for the epoch0
                eval_acc, eval_auroc, eval_eer = run_validation(model, feature_extractor, val_loader, sr=sampling_rate, use_triplet=args.use_triplet,use_mnr=args.use_mnr_loss_for_hybrid,num_negatives=args.num_negatives)
                print(f"Epoch {epoch}: Final Validation Accuracy: {eval_acc}, Final Validation AUROC: {eval_auroc}, Final Validation EER: {eval_eer}")
                os.makedirs(output_dir, exist_ok=True)
                torch.save(model.state_dict(), f'{output_dir}/{args.dataset_name}_{pretrained_model_name}_epoch_{epoch}_{args.lr}_{datetime.now()}.pth')
                
                if current_alpha_index < len(alpha_values) - 1:
                    current_alpha_index += 1
                torch.cuda.empty_cache()
            # print(f'Finetuning with margin {margin} finished')
        print(f'Finetuning with combined {args.dataset_name} finished')
        sys.exit(0)

    if args.test:
        for database_path_list in [args.data_path]:
            _, _, test_loader = get_combined_loader(database_path_list, seed, args.batch_size, use_triplet=args.use_triplet,use_mnr=args.use_mnr_loss_for_hybrid,snr_db=args.snr_db, pitch_shift=args.pitch_shift, tempo_rate=args.tempo_rate, formant_alpha=args.formant_alpha)
            model.eval()
            eval_acc, eval_auroc, eval_eer = run_validation(model, feature_extractor, test_loader, sr=sampling_rate, use_triplet=args.use_triplet,use_mnr=args.use_mnr_loss_for_hybrid,num_negatives=args.num_negatives)
            print(f"Epoch {epoch}: Final Validation Accuracy: {eval_acc}, Final Validation AUROC: {eval_auroc}, Final Validation EER: {eval_eer}")
            print('Evaluation finished')
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, evaluate, and test a Wav2Vec2BERT model.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="./ckpt", help="Directory to save checkpoints and outputs")
    parser.add_argument("--pretrained_model_name", type=str, default="wave2vec2bert", help="Name of the pretrained model")
    parser.add_argument("--data_path", type=str, nargs='+', required=True, help="List of paths to the training/testing data")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for training and evaluation")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=5e-5, help="Weight decay for the optimizer")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Number of steps between evaluations")
    parser.add_argument("--dataset_name", type=str, default="dataset", help="Name of the dataset")
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint file for testing")
    parser.add_argument("--train", type=bool, default=False, help="Flag to run training")
    parser.add_argument("--eval", type=bool, default=False, help="Flag to run evaluation")
    parser.add_argument("--test", type=bool, default=False, help="Flag to run testing")
    parser.add_argument("--is_hybrid_loss", type=bool, default=False, help="Flag to use triplet margin loss and cross entropy for training")
    parser.add_argument("--use_triplet", type=bool, default=False, help="Flag to use triplet margin loss for training")
    parser.add_argument("--margins", type=float, nargs='+', default=[2], help="List of margins for triplet loss")
    parser.add_argument("--triplet_loss_p2", type=bool, default=False, help="Flag to use triplet margin loss with distance similarity for training")
    parser.add_argument("--triplet_loss_cosine", type=bool, default=False, help="Flag to use triplet margin loss with cosine similarity for training")
    parser.add_argument("--use_mnr_loss_for_hybrid", type=bool, default=False, help="Flag to use triplet margin loss and cross entropy for training")
    parser.add_argument("--num_negatives", type=int, default=4, help="Number of negative examples for MNR Loss")
    parser.add_argument("--freeze_bert", type=bool, default=False, help="Flag to use freeze all bert layers")
    parser.add_argument("--dynamic_alpha_list", type=float, nargs='+', default=[0.30], help="List of values to give weightage towards triplet loss and ce loss")
    
    parser.add_argument("--snr_db", type=float, default=0, help="Signal-to-noise ratio for augmentation")
    parser.add_argument("--pitch_shift", type=float, default=0, help="Pitch shift factor for augmentation")
    parser.add_argument("--tempo_rate", type=float, default=0, help="Tempo adjustment factor for augmentation")
    parser.add_argument("--formant_alpha", type=float, default=0, help="Formant adjustment factor for augmentation")
    parser.add_argument("--epsilon", type=float, default=0, help="Epsilon for laplace transformation")
    args = parser.parse_args()
    main(args)



"""
Example Usage :
nohup python apm0046210-synthetic-speech-detection-train/ssd_train/train_hybrid_loss.py --data_path "../data/in_the_wild/" "../data/wavefake/" "../data/LJSpeech-1.1/" "../data/ASVspoof2021_LA_eval/" "../data/for-norm/" "../data/ASVspoof2019_LA/" --train True --batch_size 24 --use_triplet True  --triplet_loss_cosine True --is_hybrid_loss True --epochs 3 --dataset_name "combined_hybrid_alpha0.90_e3" >combined_hybrid_alpha0.90_e3.log &

"""




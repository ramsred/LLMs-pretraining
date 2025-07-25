import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import logging
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, TripletMarginLoss
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple
from dataclasses import dataclass
from pydub import AudioSegment
import librosa
import csv
from torch.autograd import Function

import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, TripletMarginLoss,TripletMarginWithDistanceLoss
import pandas as pd
import warnings

# Suppress only UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress specific warnings by message
warnings.filterwarnings("ignore", message=".*is deprecated.*")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Wav2Vec2BERT(nn.Module):
    def __init__(self, model_name, pooling_mode='mean'):
        super().__init__()
        self.num_labels = 2
        self.pooling_mode = pooling_mode
        self.wav2vec2bert = Wav2Vec2BertModel.from_pretrained(model_name)
        self.config = self.wav2vec2bert.config
        self.classifier = ClassificationHead(self.wav2vec2bert.config)

    def merged_strategy(self,hidden_states,mode="mean"):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(self,input_features,attention_mask=None,output_attentions=None,output_hidden_states=None,return_dict=None,labels=None,):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2bert(
            input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs.last_hidden_state
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.last_hidden_state,
            attentions=outputs.attentions,
        )


# ---------------- GRL -------------------
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

# ---------------- Wrapper ----------------
class Wav2Vec2BERT_GRL(nn.Module):
    def __init__(self, base: Wav2Vec2BERT, n_spk: int, lamb: float = .5):
        super().__init__()
        self.base = base
        self.lamb = lamb
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


def convert_mp3_to_wav(mp3_path):
    try:
        audio = AudioSegment.from_mp3(mp3_path)
        wav_path = mp3_path.replace('.mp3', '.wav')
        audio.export(wav_path, format='wav')
        return wav_path
    except Exception as e:
        logging.error(f"Error converting {mp3_path} to WAV: {e}")
        return None

def load_audio_file(audio_path):
    if audio_path.endswith('.mp3'):
        audio_path = convert_mp3_to_wav(audio_path)
        if audio_path is None:
            return None
    try:
        wav, _ = librosa.load(audio_path, sr=8000)
        wav = torch.tensor(wav)
        logging.info(f"Loaded audio file: {audio_path}, Shape: {wav.shape}")
        return wav
    except Exception as e:
        logging.error(f"Error loading audio file {audio_path}: {e}")
        return None


def strip_grl(grl_ckpt:str,out_file:str):
    full_state = torch.load(grl_ckpt,map_location='cpu')

    clean_state = {k:v for k,v in full_state.items() if not k.startswith("spk_head.")}

    dropped = [k for k in full_state if k.startswith("spk_head.")]

    print(f"dropped{len(dropped)} spk_head params")


    dummy_base = Wav2Vec2BERT("facebook/w2v-bert-2.0")

    dummy_wrap = Wav2Vec2BERT_GRL(dummy_base,n_spk=1)
    dummy_wrap.load_state_dict(clean_state,strict=False)
    print("Wav2Vec2BERT model architecure")
    # for name, param in model.named_parameters():
    # print("name : ",name,"param : ",param)
    task_state = {
        "wav2vec2bert":dummy_wrap.base.wav2vec2bert.state_dict(),
        "classifier" : dummy_wrap.base.classifier.state_dict(),
    }

    print("output file",out_file)

    torch.save(task_state,out_file)


# def load_model(ckpt_path, model_name, device):
#     try:
#         model = Wav2Vec2BERT(model_name)
#         if ckpt_path!='None':
#             print("ckpt_path : ",ckpt_path)
#             checkpoint = torch.load(ckpt_path, map_location=device)
#             #### Strict False can be used to remove grl layers
#             model.load_state_dict(checkpoint)
#         model = model.to(device)
#         print("Wav2Vec2BERT model architecure")
#         for name, param in model.named_parameters():
#             print("name : ",name,"param : ",param)
#         model.eval()
#         return model
#     except Exception as e:
#         logging.error(f"Error loading model: {e}")
#         return None

def load_model(ckpt_path, model_name, device):
    try:
        model = Wav2Vec2BERT(model_name)
        w = torch.load(ckpt_path,map_location=device)
        model.wav2vec2bert.load_state_dict(w['wav2vec2bert'])
        model.classifier.load_state_dict(w['classifier'])
        # if ckpt_path!='None':
        #     print("ckpt_path : ",ckpt_path)
        #     checkpoint = torch.load(ckpt_path, map_location=device)
        #     #### Strict False can be used to remove grl layers
        #     model.load_state_dict(checkpoint)
        model = model.to(device)
        print("Wav2Vec2BERT model architecure")
        for name, param in model.named_parameters():
            print("name : ",name,"param : ",param)
        model.eval()
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None


def process_audio_file(file_path, model, feature_extractor, device):
    """Process a single audio file and extract embeddings."""
    try:
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return None, None, None

        # Load audio
        wav, _ = librosa.load(file_path, sr=8000)
        wav = torch.tensor(wav).unsqueeze(0)  # Add batch dimension

        # Feature extraction
        inputs = feature_extractor(wav.numpy(), sampling_rate=16000, return_tensors="pt", padding=True).input_features.to(device)

        # Model inference
        with torch.no_grad():
            output = model(input_features=inputs)
            pooled_embeddings = model.merged_strategy(output.hidden_states, mode="mean")
            score = torch.softmax(output.logits, dim=1)[0, 1].item()
            predicted_label = 'bonafide' if score > 0.5 else 'spoof'

        return pooled_embeddings.cpu().numpy(), predicted_label, score
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return None, None, None

def visualize_pca(embeddings_2d, labels, filenames, plot_path,testset_name,finetuned_model_name):
    """
    Visualize PCA results with filenames, legend, and custom title.
    
    Parameters:
    - embeddings_2d: Numpy array of PCA embeddings (2D).
    - labels: List of integer labels (e.g., 0 or 1).
    - filenames: List of filenames corresponding to embeddings.
    - plot_path: Path to save the PCA plot.
    - testset_name: Name of the dataset used for training/testing.
    - ckpt_path: Name of the model used for generating embeddings.
    """
    try:
        # Debug: Check shapes of inputs
        logging.info(f"Embeddings shape: {embeddings_2d.shape}")
        logging.info(f"Number of labels: {len(labels)}")
        logging.info(f"Number of filenames: {len(filenames)}")

        # Ensure the number of embeddings matches the number of labels and filenames
        if len(embeddings_2d) != len(labels) or len(embeddings_2d) != len(filenames):
            raise ValueError("Mismatch between embeddings, labels, and filenames. Ensure data dimensions are consistent.")

        # Map labels to colors
        color_mapping = {1: 'green', 0: 'red'}
        colors = [color_mapping[label] for label in labels if label in color_mapping]

        # Debug: Check color mapping
        if len(colors) != len(labels):
            raise ValueError("Color mapping failed. Check if all labels are properly defined in the `color_mapping` dictionary.")

        # Create PCA plot
        plt.figure(figsize=(16, 10))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.6)
                # Add a legend for labels
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Human(Actual)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Synthetic(Actual)')
        ]

        plt.legend(handles=handles, title="Labels", loc="upper right")
        # Add title with model and dataset name

        plt.title(f"PCA Visualization\n\nModel: {finetuned_model_name} \nTestDataset: {testset_name}", fontsize=14)
        plt.xlabel("PCA Dimension 1")
        plt.ylabel("PCA Dimension 2")

        # # Annotate plot with filenames (only the filename, not the full path)
        # for i, file_path in enumerate(filenames):
        #     filename = os.path.basename(file_path)  # Extract filename from full path
        #     plt.annotate(filename, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8)

        plt.savefig(plot_path)
        logging.info(f"✅ PCA plot saved at: {plot_path}")
        plt.show()
    except Exception as e:
        logging.error(f"Error visualizing PCA: {e}")

def process_csv(input_file, output_csv, model, feature_extractor, device, plot_path, ckpt_path, finetuned_model_name,test_set=False,):
    """Process a CSV file and append results."""
    try:

        testset_name = "combined [old + new]"
        # testset_name = "test_loader"
        # Load CSV file
        if '.csv' in input_file:
            df = pd.read_csv(input_file)
        else:
            df = pd.read_excel(input_file)
        print("No of samples in test set :",len(df))
        if test_set:
            testset_name = test_set
            df = df[df['TestDataset'] == test_set]  # Filter for the desired test set

        
        if 'file_path' not in df.columns or 'label' not in df.columns:
            logging.error("The CSV file must contain 'file_path' and 'label' columns.")
            return

        # Add new columns
        df['ModelUsed'] = ckpt_path
        df['PredictedScore'] = None
        df['PredictedLabel'] = None
        df['PCA1'] = None
        df['PCA2'] = None
        df['Embeddings1024'] = None

        # Initialize lists for embeddings, labels, filenames, and failed files
        all_embeddings = []
        actual_labels = []
        successful_filenames = []  # Track files that succeeded
        failed_files = []  # Track files that failed to load

        # Process each audio file
        for index, row in df.iterrows():
            file_path = row['file_path']
            actual_label = row['label']
            # logging.info(f"Processing file: {file_path}")

            # Process audio file
            embeddings, predicted_label, predicted_score = process_audio_file(file_path, model, feature_extractor, device)
            if embeddings is not None:
                # Append results for successfully processed files
                df.at[index, 'PredictedScore'] = predicted_score
                df.at[index, 'PredictedLabel'] = predicted_label
                df.at[index, 'Embeddings1024'] = embeddings.flatten().tolist()
                all_embeddings.append(embeddings.flatten())
                actual_labels.append(actual_label)
                successful_filenames.append(file_path)  # Add filename to successful list
            else:
                # Log the failed file and continue
                logging.warning(f"Failed to process file: {file_path}")
                failed_files.append(file_path)
                df.at[index, 'PredictedScore'] = None
                df.at[index, 'PredictedLabel'] = None
                df.at[index, 'Embeddings1024'] = None

        # Log the number of failed files
        if failed_files:
            logging.warning(f"⚠️ {len(failed_files)} files failed to load and were skipped.")
            logging.warning(f"List of failed files: {failed_files}")

        # Perform PCA if embeddings were generated
        if len(all_embeddings) > 0:
            all_embeddings = np.array(all_embeddings)
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(all_embeddings)

            # Update PCA columns in the DataFrame only for successful rows
            for i, (pca1, pca2) in enumerate(embeddings_2d):
                df.at[i, 'PCA1'] = pca1
                df.at[i, 'PCA2'] = pca2

            # Visualize PCA with actual labels and successful filenames
            visualize_pca(embeddings_2d, actual_labels, successful_filenames, plot_path,testset_name,finetuned_model_name)
        else:
            logging.error("No embeddings were generated. PCA visualization skipped.")

        # Save updated CSV
        df.to_csv(output_csv, index=False)
        logging.info(f"✅ Processed CSV saved at: {output_csv}")
    except Exception as e:
        logging.error(f"Error processing CSV: {e}")

def main(args):
    """Main function."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.ckpt_path, args.model_name, device)
    if model is None:
        logging.error("Failed to load the model.")
        return

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    process_csv(args.csv_file, args.output_csv, model, feature_extractor, device, args.plot_path,args.ckpt_path,args.finetuned_model_name,args.test_set)

if __name__ == '__main__':

    is_strip_grl = True

    # models_list = ['combined_hybrid_R16_e3_wt_for-norm_ep2_20250629_210611.pth']
    # models_list = ['combined_hybrid_lambda1.0_inc_speaker_network_e3_wt_for-norm_ep2_20250628_183909.pth']
    # models_list = ['combined_hybrid_R24_e3_wt_for-norm_ep0_20250701_085626.pth']
    # models_list = ['combined_hybrid_lambda2.0_inc_speaker_network_e3_wt_for-norm_ep1_20250702_095125.pth']
    # models_list = ['combined_hybrid_lambda1.0_deep_speaker_network_e10_wt_for-norm_ep2_20250706_012944.pth']
    # models_list = ['combined_hybrid_lambda1.5_reg_speaker_network_e5_wt_for-norm_ep0_20250707_025530.pth']
    # models_list = ['combined_hybrid_lambda1.0_reg_speaker_network_e5_rep_wt_for-norm_ep0_20250707_124959.pth']
    # models_list = ['combined_hybrid_lambda_schdule_deep_speaker_network_4layer_e5_relu_b72__wt_for-norm_ep4_20250710_220639.pth']
    models_list = ['combined_hybrid_vic_schedule_e3_relu_b72__wt_for-norm_ep0_20250725_185348.pth']
    
    # out_file = "combined_hybrid_R16_e3_wt_for-norm_ep2_20250629_210611_synthetic_only.pt"
    # out_file = "combined_hybrid_lambda2.0_inc_speaker_network_e3_wt_for-norm_ep1_20250702_095125_synthetic_only.pt"
    out_file = 'combined_hybrid_vic_schedule_e3_relu_b72__wt_for-norm_ep0_20250725_185348_synthetic_only.pt'

    # Directory containing the checkpoint models
    ckpt_directory = "/home/personalai/audio_active_agent/svd_train/ckpt/"


    output_csv_files = ['combined_hybrid_lambda0.5_inc_speaker_network_e3_wt_for-norm_ep1']
    output_csv_files = [f+"_chunks_balanced_variable.csv" for f in output_csv_files]
    # Ensure models_list and output_csv_files have the same length
    if len(models_list) != len(output_csv_files):
        raise ValueError("models_list and output_csv_files must have the same length!")

    # Loop through each model and corresponding CSV
    for model, output_csv in zip(models_list, output_csv_files):
        print("model:",model)
        print("output csv : ",output_csv)
        # Construct the full path to the checkpoint file
        ckpt_path = os.path.join(ckpt_directory, model)
        # Skip processing if the model is "None"
        if model == "None":
            ckpt_path = "None"
        
        if is_strip_grl:
            out_file =  os.path.join(ckpt_directory, out_file)
            strip_grl(ckpt_path,out_file)
            ckpt_path = os.path.join(ckpt_directory, out_file)

        model_name = "facebook/w2v-bert-2.0"
        device = "cpu"
        print("ckpt_path",ckpt_path)
        # load_model(ckpt_path, model_name, device)
        # Argument parsing for each iteration
        parser = argparse.ArgumentParser(description="Process a CSV file and visualize embeddings with PCA.")
        parser.add_argument('--csv_file', type=str, required=True, help="Path to the input CSV file.")
        parser.add_argument('--output_csv', type=str, required=True, help="Path to save the output CSV file.")
        parser.add_argument('--ckpt_path', type=str, required=True, help="Path to the model checkpoint.")
        parser.add_argument('--finetuned_model_name', type=str, default="Hybrid [Triplet M=2, P=Cosine + CE]", help="Finetuned model information.")
        parser.add_argument('--model_name', type=str, default="facebook/w2v-bert-2.0", help="Name of the pre-trained model.")
        parser.add_argument('--plot_path', type=str, default="plot_test.png", help="Path to save the PCA visualization plot.")
        parser.add_argument('--test_set', type=str, default=False, help="Select the test set name old or new")

        # Add arguments dynamically based on the current model and output_csv
        args = parser.parse_args([
            '--csv_file', 'Inputmeta_balanced_variable_chunks_06_04.xlsx',  # Replace with your input CSV file
            '--output_csv', f"/home/personalai/audio_active_agent/svd_train/apm0046210-synthetic-speech-detection-train/ssd_train/csv_output/{output_csv}",
            '--ckpt_path', ckpt_path
        ])

        # Call the main function for processing
        # main(args)













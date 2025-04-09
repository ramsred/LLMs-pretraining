import os
import random
import librosa
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


class Wav2Vec2BertForContrastiveLearning(nn.Module):
    def __init__(self, model_name, pooling_mode='mean', normalize_embeddings=True):
        super().__init__()
        self.pooling_mode = pooling_mode
        self.normalize_embeddings = normalize_embeddings
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.wav2vec2bert = Wav2Vec2BertModel.from_pretrained(model_name)
        self.config = self.wav2vec2bert.config

    def merged_strategy(self, hidden_states, mode="mean"):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs, _ = torch.max(hidden_states, dim=1)
        else:
            raise ValueError(
                f"Undefined pooling mode: '{mode}'. Choose from ['mean', 'sum', 'max']."
            )
        return outputs

    def preprocess_audio(self, audio_waveforms, sampling_rate=16000):
        """
        Helper function to preprocess raw audio waveforms directly.
        This method converts raw audio to features expected by Wav2Vec2BERT.
        """
        inputs = self.feature_extractor(
            audio_waveforms, 
            sampling_rate=sampling_rate, 
            return_tensors="pt", 
            padding=True
        )
        return inputs['input_values']

    def forward(self, input_features):
        outputs = self.wav2vec2bert(input_features)
        hidden_states = outputs.last_hidden_state
        embeddings = self.merged_strategy(hidden_states, mode=self.pooling_mode)

        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)  # Essential for triplet loss stability

        return embeddings
def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds

def compute_eer(ground_truth, predictions):
    """
    Expecting ground_truth and predictions to be numpy arrays of the same length;
    Defining deepfakes (ground_truth == 1) as target scores and bonafide (ground_truth == 0) as nontarget scores.
    """
    assert ground_truth.shape == predictions.shape, "ground_truth and predictions must have the same shape"
    assert len(ground_truth.shape) == 1, "ground_truth and predictions must be 1D arrays"

    target_scores = predictions[ground_truth == 1]
    nontarget_scores = predictions[ground_truth == 0]

    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]

def run_validation(model, feature_extractor, data_loader, sr):

    outputs_list = []
    labels_list = []
    train_loss = []
    num_total = 0

    model.eval()

    with torch.no_grad():
        for batch_x, batch_y, name in tqdm(data_loader, desc="Evaluating"):
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.numpy()
            inputs = feature_extractor(batch_x, sampling_rate=sr, return_attention_mask=True, padding_value=0, return_tensors="pt").to(device)
            batch_y = batch_y.to(device)
            inputs['labels'] = batch_y
            outputs = model(**inputs)
            train_loss.append(outputs.loss.item())
            batch_probs = outputs.logits.softmax(dim=-1)
            batch_label = batch_y.detach().to('cpu').numpy().tolist()
            outputs_list.extend(batch_probs[:, 1].tolist())
            labels_list.extend(batch_label)

        auroc = roc_auc_score(labels_list, outputs_list)
        eer = compute_eer(np.array(labels_list), np.array(outputs_list))
        preds = (np.array(outputs_list) > eer[1]).astype(int)
        acc = np.mean(np.array(labels_list) == np.array(preds))
        prec = precision_score(labels_list, preds)
        recall = recall_score(labels_list, preds)
        f1 = f1_score(labels_list, preds)
        print(f'Validation Accuracy: {acc} \t F1: {f1} \t Precision: {prec} \t Recall: {recall}, AUROC: {auroc} \t EER: {eer}')
        
    return acc, auroc, eer

def pad(x, max_len=64000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]

    pad_length = max_len - x_len
    padded_x = np.concatenate([x, np.zeros(pad_length)], axis=0)
    return padded_x


def seed_worker(worker_id):
    """
    Used in generating seed for the worker of torch.utils.data.Dataloader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class AudioDataset(Dataset):
    def __init__(self, list_IDs, labels, transform=False):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.cut = 64000
        self.transform = transform

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, sr = librosa.load(str(key), sr=8000)
        X = pad(X, max_len=self.cut)
        y = self.labels[key]

        return X, y, key


class CombinedTripletDataset(Dataset):
    def __init__(self, train_ids, train_labels, sr=16000, max_len=64000):
        super().__init__()
        self.sr = sr
        self.max_len = max_len
        
        # Separate real/fake paths
        self.real_paths = [path for path in train_ids if train_labels[path] == 1]
        self.fake_paths = [path for path in train_ids if train_labels[path] == 0]

    def __len__(self):
        return len(self.real_paths)

    def pad(self, x):
        if len(x) >= self.max_len:
            return x[:self.max_len]
        else:
            pad_len = self.max_len - len(x)
            return np.concatenate([x, np.zeros(pad_len)])

    def __getitem__(self, idx):
        anchor_path = self.real_paths[idx]
        anchor_waveform, _ = librosa.load(anchor_path, sr=self.sr)
        anchor_waveform = self.pad(anchor_waveform)

        # Positive sample (ensure it's different from anchor)
        pos_path = random.choice(self.real_paths)
        while pos_path == anchor_path:
            pos_path = random.choice(self.real_paths)
        pos_waveform, _ = librosa.load(pos_path, sr=self.sr)
        pos_waveform = self.pad(pos_waveform)

        # Negative sample
        neg_path = random.choice(self.fake_paths)
        neg_waveform, _ = librosa.load(neg_path, sr=self.sr)
        neg_waveform = self.pad(neg_waveform)

        return (
            torch.tensor(anchor_waveform, dtype=torch.float32),
            torch.tensor(pos_waveform, dtype=torch.float32),
            torch.tensor(neg_waveform, dtype=torch.float32)
        )


def genIn_the_wild_list_new(database_path: str):

    import csv
    database_path = database_path
    file = os.path.join(database_path, 'meta.csv')
    d_meta = {}
    file_list = []
    data_list0 = []
    data_list1 = []

    with open(file, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for line in csv_reader:
            key, name, label = line
    
            if label == 'bona-fide':
                data_list1.append(os.path.join(database_path,key))
                d_meta[os.path.join(database_path, key)] = 1
            else:
                data_list0.append(os.path.join(database_path, key))
                d_meta[os.path.join(database_path, key)] = 0
    
    file_list = data_list0 + data_list1
    return d_meta, file_list

def getASVSpoof2021_list_new(data_path):

    d_meta = {}
    file_list = []
    file_path = os.path.join(data_path,'ASVspoof2021.LA.cm.eval.trl_updated.txt')
    with open(file_path, "r") as f:
        l_meta = f.readlines()
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            key = os.path.join(data_path,'flac', key + '.flac')
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0

    return d_meta, file_list

def getASVSpoof2019_list_new(data_path):
    d_meta = {}
    file_list = []

    def read_meta_file(meta_file, audio_folder):
        with open(meta_file, "r") as f:
            l_meta = f.readlines()
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            key = os.path.join(audio_folder, key + '.flac')
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0

    # Define the metadata files and corresponding audio folders for train, validation, and test sets
    datasets = [
        ('ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trl.txt', 'ASVspoof2019_LA_train/flac'),
        ('ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.val.trl.txt', 'ASVspoof2019_LA_val/flac'),
        ('ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.test.trl.txt', 'ASVspoof2019_LA_test/flac')
    ]

    # Loop through each dataset and read the metadata files
    for meta_file, audio_folder in datasets:
        meta_file_path = os.path.join(data_path, meta_file)
        audio_folder_path = os.path.join(data_path, audio_folder)
        read_meta_file(meta_file_path, audio_folder_path)

    return d_meta, file_list

def genLJSpeech_list_new(data_path):
    d_meta = {}
    data_list = []

    # Get LJSpeech
    real_datapath = os.path.join(data_path, 'wavs/')
    file_list = os.listdir(real_datapath)

    for line in file_list:
        key = os.path.join(real_datapath, line)
        data_list.append(key)
        d_meta[key] = 1

    return d_meta, data_list

def genWavefake_list_new(data_path):
    d_meta = {}
    data_list0 = []
    data_list1 = []

    ## Get wavefake
    folders = ['ljspeech_melgan',
               'ljspeech_parallel_wavegan',
               'ljspeech_multi_band_melgan',
               'ljspeech_full_band_melgan',
               'ljspeech_waveglow',
               'ljspeech_hifiGAN']

    for i in range(len(folders)):
        file_list = os.listdir(os.path.join(data_path, folders[i]))
        for line in file_list:
            key = os.path.join(data_path, folders[i], line)
            data_list0.append(key)
            d_meta[key] = 0
    return d_meta, data_list0


def gen_for_norm_list_new(data_path):
    def read_data(subfolder):
        ids = []
        d_meta = {}
        categories = ['fake', 'real']

        for category in categories:
            folder_path = os.path.join(data_path, subfolder, category)
            file_list = os.listdir(folder_path)
            for file_name in file_list:
                key = os.path.join(folder_path, file_name)
                ids.append(key)
                if category == 'fake':
                    d_meta[key]=0
                else:
                    d_meta[key]=1
        
        return d_meta, ids

    # Read each dataset separately
    train_meta, train_ids = read_data('training')
    val_meta, val_ids = read_data('validation')
    test_meta, test_ids = read_data('testing')

    return train_ids,train_meta,val_ids,val_meta,test_ids,test_meta


def split_dict(file_dict, train_ratio=0.70, val_ratio=0.10, test_ratio=0.20, seed=None):
    # Ensure the ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"
    
    # Set the random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Separate filenames based on labels
    label_1_files = [filename for filename, label in file_dict.items() if label == 1]
    label_0_files = [filename for filename, label in file_dict.items() if label == 0]

    # Shuffle the files
    random.shuffle(label_1_files)
    random.shuffle(label_0_files)

    # Calculate the number of files for each split
    num_label_1_files = len(label_1_files)
    num_label_0_files = len(label_0_files)

    train_size_1 = int(train_ratio * num_label_1_files)
    val_size_1 = int(val_ratio * num_label_1_files)
    test_size_1 = num_label_1_files - train_size_1 - val_size_1

    train_size_0 = int(train_ratio * num_label_0_files)
    val_size_0 = int(val_ratio * num_label_0_files)
    test_size_0 = num_label_0_files - train_size_0 - val_size_0

    # Split the files for label 1
    train_files_1 = label_1_files[:train_size_1]
    val_files_1 = label_1_files[train_size_1:train_size_1 + val_size_1]
    test_files_1 = label_1_files[train_size_1 + val_size_1:]

    # Split the files for label 0
    train_files_0 = label_0_files[:train_size_0]
    val_files_0 = label_0_files[train_size_0:train_size_0 + val_size_0]
    test_files_0 = label_0_files[train_size_0 + val_size_0:]

    # Combine label 1 and label 0 files for each split
    train_files = train_files_1 + train_files_0
    val_files = val_files_1 + val_files_0
    test_files = test_files_1 + test_files_0

    # Create the split dictionaries
    train_dict = {filename: file_dict[filename] for filename in train_files}
    val_dict = {filename: file_dict[filename] for filename in val_files}
    test_dict = {filename: file_dict[filename] for filename in test_files}

    return train_dict, val_dict, test_dict

def combine_all_dicts(meta, train_labels, train_ids, val_labels, val_ids, test_labels, test_ids, seed=None):
    train_dict, val_dict, test_dict = split_dict(meta, seed=seed)

    # Shuffle the training, validation, and test sets
    train_keys = list(train_dict.keys())
    val_keys = list(val_dict.keys())0
    test_keys = list(test_dict.keys())

    if seed is not None:
        random.seed(seed)

    random.shuffle(train_keys)
    random.shuffle(val_keys)
    random.shuffle(test_keys)

    train_ids += train_keys
    val_ids += val_keys
    test_ids += test_keys

    train_labels.update(train_dict)
    val_labels.update(val_dict)
    test_labels.update(test_dict)

    return train_labels, train_ids, val_labels, val_ids, test_labels, test_ids

def gen_combined_list(data_paths, seed=None):
    train_labels, val_labels, test_labels = {}, {}, {}
    train_ids, val_ids, test_ids = [], [], []

    for data_path in data_paths:
        if data_path == "../data/in_the_wild/":
            meta, data_list = genIn_the_wild_list_new(data_path)
            train_labels, train_ids, val_labels, val_ids, test_labels, test_ids = \
                combine_all_dicts(meta, train_labels, train_ids, val_labels, val_ids, test_labels, test_ids, seed)
        elif data_path == "../data/ASVspoof2021_LA_eval/":
            meta, data_list = getASVSpoof2021_list_new(data_path)
            train_labels, train_ids, val_labels, val_ids, test_labels, test_ids = \
                combine_all_dicts(meta, train_labels, train_ids, val_labels, val_ids, test_labels, test_ids, seed)
        elif data_path == "../data/wavefake/":
            meta, data_list = genWavefake_list_new(data_path)
            train_labels, train_ids, val_labels, val_ids, test_labels, test_ids = \
                combine_all_dicts(meta, train_labels, train_ids, val_labels, val_ids, test_labels, test_ids, seed)
        elif data_path == "../data/LJSpeech-1.1/":
            meta, data_list = genLJSpeech_list_new(data_path)
            train_labels, train_ids, val_labels, val_ids, test_labels, test_ids = \
                combine_all_dicts(meta, train_labels, train_ids, val_labels, val_ids, test_labels, test_ids, seed)
        elif data_path =="../data/for-norm/":
        
            train_keys,train_meta,val_keys,val_meta,test_keys,test_meta = gen_for_norm_list_new(data_path)
            train_ids += train_keys
            val_ids += val_keys
            test_ids += test_keys

            train_labels.update(train_meta)
            val_labels.update(val_meta)
            test_labels.update(test_meta)
        elif data_path =="../data/for-rerecorded/":
            train_keys,train_meta,val_keys,val_meta,test_keys,test_meta = gen_for_norm_list_new(data_path)
            train_ids += train_keys
            val_ids += val_keys
            test_ids += test_keys

            train_labels.update(train_meta)
            val_labels.update(val_meta)
            test_labels.update(test_meta)

        elif data_path =="../data/ASVspoof2019_LA/":
            meta, data_list = getASVSpoof2019_list_new(data_path)
            train_labels, train_ids, val_labels, val_ids, test_labels, test_ids = \
                combine_all_dicts(meta, train_labels, train_ids, val_labels, val_ids, test_labels, test_ids, seed)
        else:
            print("Data path not found.")
            
    return train_labels, train_ids, val_labels, val_ids, test_labels, test_ids

def get_combined_loader(database_path_list: str, seed: int, batch_size: int):
    train_labels, train_ids, val_labels, val_ids, test_labels, test_ids = gen_combined_list(database_path_list,seed)
 
    # Create instances of the AudioDataset
    train_dataset = AudioDataset(train_ids, train_labels)
    val_dataset = AudioDataset(val_ids, val_labels)
    test_dataset = AudioDataset(test_ids, test_labels)

    gen = torch.Generator()
    gen.manual_seed(seed)

    # Create the training DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, worker_init_fn=seed_worker, generator=gen)

    # Create the validation DataLoader
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, worker_init_fn=seed_worker, generator=gen)

    # Create the test DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, worker_init_fn=seed_worker, generator=gen)

    return train_loader, val_loader, test_loader


def get_combined_loader_cl(database_path_list, seed, batch_size):
    train_labels, train_ids, val_labels, val_ids, test_labels, test_ids = gen_combined_list(database_path_list, seed)

    train_dataset = CombinedTripletDataset(train_ids, train_labels)
    val_dataset = CombinedTripletDataset(val_ids, val_labels)
    test_dataset = CombinedTripletDataset(test_ids, test_labels)

    gen = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        worker_init_fn=seed_worker, generator=gen, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        worker_init_fn=seed_worker, generator=gen, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        worker_init_fn=seed_worker, generator=gen, drop_last=True
    )

    return train_loader, val_loader, test_loader

def get_generator(test_ids,test_labels,seed:int,batch_size:int):
     # Create instances of the AudioDataset
    test_dataset = AudioDataset(test_ids, test_labels)

    gen = torch.Generator()
    gen.manual_seed(seed)
   
    # Create the test DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, worker_init_fn=seed_worker, generator=gen)
    return test_loader

def get_test_loader(data_paths: str, seed: int, batch_size: int):
    test_labels = {}
    test_ids = []
    test_loader = None
    for data_path in data_paths:
        if data_path == "../data/in_the_wild/":
            meta, data_list = genIn_the_wild_list_new(data_path)
            test_labels = test_labels | meta
            test_ids += data_list
           
        elif data_path == "../data/ASVspoof2021_LA_eval/":
            meta, data_list = getASVSpoof2021_list_new(data_path)
            test_labels = test_labels | meta
            test_ids += data_list

        elif data_path == "../data/wavefake/":
            meta_0, data_list0 = genWavefake_list_new(data_path)
            test_labels = test_labels | meta_0
            test_ids += data_list0
            
        elif data_path == "../data/LJSpeech-1.1/" :
            meta_1, data_list1 = genLJSpeech_list_new(data_path)
            test_labels = test_labels | meta_1
            test_ids += data_list1
            
        elif data_path =="../data/for-norm/":
            meta, data_list = gen_for_norm_list(data_path)
            test_labels = test_labels | meta
            test_ids += data_list
 
        elif data_path =="../data/for-rerecorded/":
            meta, data_list = gen_for_norm_list(data_path)
            test_labels = test_labels | meta
            test_ids += data_list
    
        elif data_path =="../data/ASVspoof2019_LA/":
            meta, data_list = getASVSpoof2019_list_new(data_path)
            test_labels = test_labels | meta
            test_ids += data_list
            
        else:
            print("Data path not found.")
 
    test_loader = get_generator(test_ids,test_labels,seed,batch_size)
    return test_loader



import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
import os
from datetime import datetime
import argparse
from models.wave2vec2bert import Wav2Vec2BertForContrastiveLearning
from data_utils import get_combined_loader_cl
import numpy as np
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

def run_validation(model, val_loader, feature_extractor, sampling_rate=16000):
    model.eval()
    val_loss = 0.0
    criterion = nn.TripletMarginLoss(margin=1.0)

    with torch.no_grad():
        for anchor_wave, pos_wave, neg_wave in val_loader:

            # Preprocess inputs
            # anchor_inputs = feature_extractor(anchor_wave.numpy(), sampling_rate=sampling_rate, return_tensors="pt", padding=True).input_values.to(device)
            # pos_inputs = feature_extractor(pos_wave.numpy(), sampling_rate=sampling_rate, return_tensors="pt", padding=True).input_values.to(device)
            # neg_inputs = feature_extractor(neg_wave.numpy(), sampling_rate=sampling_rate, return_tensors="pt", padding=True).input_values.to(device)
            
            anchor_inputs = feature_extractor(
                anchor_wave.numpy(), 
                sampling_rate=sampling_rate, 
                return_tensors="pt", 
                padding=True
            ).input_features.to(device)

            pos_inputs = feature_extractor(
                pos_wave.numpy(), 
                sampling_rate=sampling_rate, 
                return_tensors="pt", 
                padding=True
            ).input_features.to(device)

            neg_inputs = feature_extractor(
                neg_wave.numpy(), 
                sampling_rate=sampling_rate, 
                return_tensors="pt", 
                padding=True
            ).input_features.to(device)

            anchor_emb = model(anchor_inputs)
            pos_emb = model(pos_inputs)
            neg_emb = model(neg_inputs)

            loss = criterion(anchor_emb, pos_emb, neg_emb)
            val_loss += loss.item() * anchor_wave.size(0)

    avg_loss = val_loss / len(val_loader.dataset)
    print(f"[Validation] Avg Loss: {avg_loss:.4f}")
    model.train()

def train(model, trn_loader, val_loader, epochs, lr, weight_decay, eval_steps, output_dir, dataset_name, pretrained_model_name, sampling_rate=16000):
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.TripletMarginLoss(margin=1.0)

    model.train()
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

    for epoch in range(epochs):
        epoch_loss, num_total = 0.0, 0

        for step, (anchor_wave, pos_wave, neg_wave) in enumerate(tqdm(trn_loader, desc=f"Epoch {epoch+1}/{epochs}")):

            anchor_inputs = feature_extractor(anchor_wave.numpy(), sampling_rate=sampling_rate, return_tensors="pt", padding=True).input_features.to(device)
            pos_inputs = feature_extractor(pos_wave.numpy(), sampling_rate=sampling_rate, return_tensors="pt", padding=True).input_features.to(device)
            neg_inputs = feature_extractor(neg_wave.numpy(), sampling_rate=sampling_rate, return_tensors="pt", padding=True).input_features.to(device)

            optim.zero_grad()

            anchor_emb = model(anchor_inputs)
            pos_emb = model(pos_inputs)
            neg_emb = model(neg_inputs)

            loss = criterion(anchor_emb, pos_emb, neg_emb)
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * anchor_wave.size(0)
            num_total += anchor_wave.size(0)

            if (step + 1) % eval_steps == 0:
                print(f"[Epoch {epoch+1}, Step {step+1}] Loss: {loss.item():.4f}")
                run_validation(model, val_loader, feature_extractor, sampling_rate)

        avg_epoch_loss = epoch_loss / num_total
        print(f"[Epoch {epoch+1}] Avg Loss: {avg_epoch_loss:.4f}")

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_path = os.path.join(
            output_dir,
            f'{dataset_name}_{pretrained_model_name}_epoch_{epoch+1}_{timestamp}.pth'
        )
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.pretrained_model_name == 'wave2vec2bert':
        model_name = "facebook/w2v-bert-2.0"
        model = Wav2Vec2BertForContrastiveLearning(model_name).to(device)
    else:
        raise ValueError(f"Model {args.pretrained_model_name} not supported")

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    if args.train:
        train_loader, val_loader, _ = get_combined_loader_cl(args.data_path, args.seed, args.batch_size)
        train(model, train_loader, val_loader, args.epochs, args.lr, args.weight_decay, args.eval_steps, args.output_dir, args.dataset_name, args.pretrained_model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_dir", type=str, default="./ckpt")
    parser.add_argument("--pretrained_model_name", type=str, default="wave2vec2bert")
    parser.add_argument("--data_path", type=str, nargs='+', required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--dataset_name", type=str, default="dataset")
    parser.add_argument("--train", default='False')

    args = parser.parse_args()
    main(args)

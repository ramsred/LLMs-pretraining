import os
import random
import librosa
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pyworld as pw
import soundfile as sf

import warnings
warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"

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

def run_validation(model, feature_extractor, data_loader, sr,use_triplet=False,use_mnr=False,num_negatives=4):
    outputs_list = []
    labels_list = []
    train_loss = []
    num_total = 0

    model.eval()

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            if use_triplet:
                anchor, positive, negative, batch_y = batch
                # print(f"Anchor shape: {anchor.shape}")
                # print(f"Positive shape: {positive.shape}")
                # print(f"Negative shape: {negative.shape}")
                if use_mnr and negative.shape[0] == anchor.shape[0] and negative.shape[1] == num_negatives:
                    # MNR loss: Negative contains multiple negatives per anchor
                    # print("Using MNR loss")
                    negative = negative.view(-1, negative.size(-1))  # Reshape negatives for MNR

                batch_x = torch.cat((anchor, positive, negative), dim=0)

                batch_y_negatives = torch.zeros(negative.size(0), dtype=batch_y.dtype, device=batch_y.device)  # Negative labels (0 for MNR)
                # batch_y = torch.cat((batch_y, batch_y, torch.zeros_like(batch_y)), dim=0)  # Combine labels: 1 for anchor and positive, 0 for negative
                batch_y = torch.cat((batch_y, batch_y,batch_y_negatives), dim=0)
            else:
                batch_x, batch_y,_ = batch

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

# Laplace Mechanism Function
def laplace_mechanism(features, sensitivity, epsilon,random_seed=None):
    """
    Add Laplace noise to features for differential privacy.
    Args:
        features (np.array): Input features to be privatized.
        sensitivity (float): Sensitivity of the feature set.
        epsilon (float): Privacy budget.
    Returns:
        np.array: Noisy features with Laplace noise added.
    """
        # Set the random seed if provided (for reproducibility)
    if random_seed is not None:
        np.random.seed(random_seed)

    scale = sensitivity / epsilon  # Scale parameter for Laplace distribution
    noise = np.random.laplace(0, scale, features.shape)  # Generate Laplace noise
    noisy_features = features + noise  # Add noise to features
    return noisy_features

# Function to add Gaussian noise
def add_gaussian_noise(audio, sr, snr_db,random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)  # Set seed for reproducibility
    rms_audio = np.sqrt(np.mean(audio**2))
    rms_noise = rms_audio / (10**(snr_db / 20))
    noise = np.random.normal(0, rms_noise, audio.shape[0])
    audio_noisy = audio + noise
    return np.clip(audio_noisy, -1.0, 1.0)

# Function to apply pitch shifting
def apply_pitch_shift(audio, sr, pitch_shift):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)

# Function to apply tempo stretching
def apply_tempo_stretch(audio, tempo_rate):
    return librosa.effects.time_stretch(audio, rate=tempo_rate)

# Function to apply formant shifting
def shift_formants(audio, sr, alpha=1.0):
    # Extract features using pyworld
    _f0, t = pw.harvest(audio, sr)
    sp = pw.cheaptrick(audio, _f0, t, sr)
    ap = pw.d4c(audio, _f0, t, sr)

    # Stretch spectral envelope (formant shifting)
    sp_alpha = np.zeros_like(sp)
    for i in range(len(sp)):
        interpolated = np.interp(
            np.arange(0, sp.shape[1], alpha),
            np.arange(0, sp.shape[1]),
            sp[i]
        )
        sp_alpha[i, :] = np.resize(interpolated, sp.shape[1])
    
    # Re-synthesize the audio
    return pw.synthesize(_f0, sp_alpha, ap, sr)


def apply_transformations(waveform,pitch_shift=0,tempo_rate=0,formant_alpha=0,snr_db=0):
    """
    Pads the waveform to a fixed length and applies transformations based on flags.
    """

    # Apply pitch shift if specified
    if pitch_shift!=0:
        # print("pitch shift applied")
        waveform = apply_pitch_shift(waveform, sr=8000, pitch_shift=pitch_shift)
    
    # Apply tempo stretch if specified
    if tempo_rate!=0:
        # print("tempo rate applied")
        waveform = apply_tempo_stretch(waveform, tempo_rate=tempo_rate)
    
    # Apply formant shifting if specified
    if formant_alpha!=0:
        # print("formant alpha applied")
        waveform = shift_formants(waveform, sr=8000, alpha=formant_alpha)
    
    # Apply Gaussian noise if specified
    if snr_db!=0:
        # print("Gaussian noise applied")
        waveform = add_gaussian_noise(waveform, sr=8000, snr_db=snr_db,random_seed=1234)
    
    return waveform

# Updated TripletDataset with data augmentation techniques
class TripletDataset(Dataset):
    def __init__(self, list_IDs, labels, transform=False, use_mnr=False, num_negatives=4,
                 snr_db=0, pitch_shift=0, tempo_rate=0, formant_alpha=0):
        """
        Args:
        - list_IDs: List of audio file paths.
        - labels: Dictionary mapping file paths to labels (1 for real, 0 for fake).
        - transform: Whether to apply transformation (default=False).
        - use_mnr: Flag to enable Multiple Negative Ranking (MNR) loss.
        - num_negatives: Number of negative samples for MNR loss (default=5).
        - snr_db: Signal-to-noise ratio for Gaussian noise addition (default=None, no noise added if None).
        - pitch_shift: Pitch shift in semitones (default=None, no pitch shift if None).
        - tempo_rate: Tempo stretch rate (default=None, no tempo stretch if None).
        - formant_alpha: Scaling factor for formant shifting (default=None, no formant shift if None).
        """
        self.list_IDs = list_IDs
        self.labels = labels
        self.cut = 64000  # Default max length for padding (e.g., 2 seconds at 8 kHz)
        self.transform = transform
        self.use_mnr = use_mnr
        self.num_negatives = num_negatives
        self.snr_db = snr_db
        self.pitch_shift = pitch_shift
        self.tempo_rate = tempo_rate
        self.formant_alpha = formant_alpha

        # Separate real/fake paths
        self.real_paths = [path for path in list_IDs if labels[path] == 1]
        self.fake_paths = [path for path in list_IDs if labels[path] == 0]

    def __len__(self):
        return len(self.real_paths)

    def __getitem__(self, index):
        # Anchor sample
        anchor_path = self.real_paths[index]
        anchor_waveform, _ = librosa.load(str(anchor_path), sr=8000)
        anchor_waveform = pad(anchor_waveform, max_len=self.cut)
        anchor_waveform = apply_transformations(anchor_waveform,snr_db=self.snr_db, pitch_shift=self.pitch_shift, tempo_rate=self.tempo_rate, formant_alpha=self.formant_alpha)

        # Positive sample (ensure it's different from anchor)
        pos_path = random.choice(self.real_paths)
        while pos_path == anchor_path:
            pos_path = random.choice(self.real_paths)
        pos_waveform, _ = librosa.load(str(pos_path), sr=8000)
        pos_waveform = pad(pos_waveform, max_len=self.cut)
        pos_waveform = apply_transformations(pos_waveform,snr_db=self.snr_db, pitch_shift=self.pitch_shift, tempo_rate=self.tempo_rate, formant_alpha=self.formant_alpha)

        # Negative samples
        if self.use_mnr:
            neg_waveforms = []
            if len(self.fake_paths) > 0:
                neg_paths = random.sample(self.fake_paths, min(self.num_negatives, len(self.fake_paths)))
                for neg_path in neg_paths:
                    try:
                        neg_waveform, _ = librosa.load(str(neg_path), sr=8000)
                        neg_waveform = pad(neg_waveform, max_len=self.cut)
                        neg_waveform = apply_transformations(neg_waveform,snr_db=self.snr_db, pitch_shift=self.pitch_shift, tempo_rate=self.tempo_rate, formant_alpha=self.formant_alpha)
                        neg_waveforms.append(torch.tensor(neg_waveform))
                    except Exception as e:
                        print(f"Error loading negative path {neg_path}: {e}")
                neg_waveform = torch.stack(neg_waveforms)
            else:
                print("Warning: No fake paths available for MNR negatives!")
        else:
            if len(self.fake_paths) > 0:
                neg_path = random.choice(self.fake_paths)
                try:
                    neg_waveform, _ = librosa.load(str(neg_path), sr=8000)
                    neg_waveform = pad(neg_waveform, max_len=self.cut)
                    neg_waveform = apply_transformations(neg_waveform,snr_db=self.snr_db, pitch_shift=self.pitch_shift, tempo_rate=self.tempo_rate, formant_alpha=self.formant_alpha)
                except Exception as e:
                    print(f"Error loading negative path {neg_path}: {e}")
            else:
                print("Warning: No fake paths available for negative sample!")

        # Return anchor, positive, negatives, and label
        y = self.labels[anchor_path]
        return anchor_waveform, pos_waveform, neg_waveform, y

# Updated get_combined_loader with MNR Loss Support
def get_combined_loader(database_path_list, seed, batch_size, use_triplet=False, use_mnr=False, num_negatives=4,snr_db=None, pitch_shift=None, tempo_rate=None, formant_alpha=None):
    """
    Args:
    - database_path_list: List of paths to the databases.
    - seed: Random seed for reproducibility.
    - batch_size: Batch size for DataLoader.
    - use_triplet: Enable triplet loss (default=False).
    - use_mnr: Enable MNR loss (default=False).
    - num_negatives: Number of negatives for MNR loss (default=5).
    """
    train_labels, train_ids, val_labels, val_ids, test_labels, test_ids = gen_combined_list(database_path_list, seed)

    if use_triplet or use_mnr:  # Enable Triplet or MNR loss
        train_dataset = TripletDataset(train_ids, train_labels, use_mnr=use_mnr, num_negatives=num_negatives,snr_db=snr_db, pitch_shift=pitch_shift, tempo_rate=tempo_rate, formant_alpha=formant_alpha)
        val_dataset = TripletDataset(val_ids, val_labels, use_mnr=use_mnr, num_negatives=num_negatives,snr_db=snr_db, pitch_shift=pitch_shift, tempo_rate=tempo_rate, formant_alpha=formant_alpha)
        test_dataset = TripletDataset(test_ids, test_labels, use_mnr=use_mnr, num_negatives=num_negatives,snr_db=snr_db, pitch_shift=pitch_shift, tempo_rate=tempo_rate, formant_alpha=formant_alpha)
    else:
        train_dataset = AudioDataset(train_ids, train_labels)
        val_dataset = AudioDataset(val_ids, val_labels)
        test_dataset = AudioDataset(test_ids, test_labels)


        # Log counts for real and fake samples
    # real_train, fake_train = train_dataset.get_sample_counts()
    # real_val, fake_val = val_dataset.get_sample_counts()
    # real_test, fake_test = test_dataset.get_sample_counts()

    # print(f"Train Dataset - Real Samples: {real_train}, Fake Samples: {fake_train}")
    # print(f"Validation Dataset - Real Samples: {real_val}, Fake Samples: {fake_val}")
    # print(f"Test Dataset - Real Samples: {real_test}, Fake Samples: {fake_test}")
 

    gen = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, worker_init_fn=seed_worker, generator=gen, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, worker_init_fn=seed_worker, generator=gen, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, worker_init_fn=seed_worker, generator=gen, drop_last=True)

    return train_loader, val_loader, test_loader

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
    val_keys = list(val_dict.keys())
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
            print("Data path not found.",data_path)
            
    return train_labels, train_ids, val_labels, val_ids, test_labels, test_ids
   
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
            print("Data path not found.",data_path)
 
    test_loader = get_generator(test_ids,test_labels,seed,batch_size)
    return test_loader

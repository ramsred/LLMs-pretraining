import os
import csv
import random

import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import AutoFeatureExtractor, AutoModel

###############################################################################
# 1. Parse meta.csv => produce (file_list, label_dict)
###############################################################################
def gen_in_the_wild_list_new(database_path: str):
    """
    Reads release_in_the_wild/meta.csv:
      - Expects a header, skip it
      - lines: key, name, label
        'bona-fide' => label=1
        else => label=0
    Returns:
      d_meta   = dict {full_wav_path -> 0 or 1}
      file_list = all wav paths (spoofs first, then bona-fide)
    """
    file = os.path.join(database_path, 'meta.csv')
    d_meta = {}
    data_list0 = []  # label=0
    data_list1 = []  # label=1

    with open(file, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # skip header row

        for row in csv_reader:
            # example row: [key, speaker, label]
            key, speaker, label = row
            full_path = os.path.join(database_path, key)
            if label.strip().lower() == 'bona-fide':
                data_list1.append(full_path)
                d_meta[full_path] = 1
            else:
                data_list0.append(full_path)
                d_meta[full_path] = 0

    file_list = data_list0 + data_list1
    return d_meta, file_list

###############################################################################
# 2. Triplet Dataset for (anchor, positive, negative)
###############################################################################
class InTheWildTripletDataset(Dataset):
    """
    Yields (anchor_waveform, positive_waveform, negative_waveform) for each index.
    - anchor & positive share the same label
    - negative is from the other label
    Loads audio with librosa at 16 kHz (mono).
    """
    def __init__(self, database_path, sr=8000, transform=None):
        super().__init__()
        self.database_path = database_path
        self.sr = sr
        self.transform = transform

        # parse meta.csv => get d_meta & file_list
        self.d_meta, self.file_list = gen_in_the_wild_list_new(database_path)

        # separate indices by label
        self.indices_label0 = []  # spoof
        self.indices_label1 = []  # bona-fide

        for i, fpath in enumerate(self.file_list):
            lbl = self.d_meta[fpath]  # 0 or 1
            if lbl == 0:
                self.indices_label0.append(i)
            else:
                self.indices_label1.append(i)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # anchor
        anchor_path = self.file_list[idx]
        anchor_label = self.d_meta[anchor_path]
        anchor_waveform = self._load_audio(anchor_path)

        # positive (same label)
        if anchor_label == 0:
            pos_idx = random.choice(self.indices_label0)
        else:
            pos_idx = random.choice(self.indices_label1)
        pos_path = self.file_list[pos_idx]
        pos_waveform = self._load_audio(pos_path)

        # negative (opposite label)
        if anchor_label == 0:
            neg_idx = random.choice(self.indices_label1)
        else:
            neg_idx = random.choice(self.indices_label0)
        neg_path = self.file_list[neg_idx]
        neg_waveform = self._load_audio(neg_path)

        return anchor_waveform, pos_waveform, neg_waveform

    def _load_audio(self, wav_path):
        """
        Loads single-channel audio at self.sr using librosa, returns torch FloatTensor [time].
        """
        waveform_np, sr_ = librosa.load(wav_path, sr=self.sr, mono=True)
        waveform_torch = torch.from_numpy(waveform_np)

        if self.transform:
            waveform_torch = self.transform(waveform_torch)

        return waveform_torch

###############################################################################
# 3. Collate function => pad waveforms
###############################################################################
def triplet_collate_fn(batch):
    """
    batch: list of (anchor_wave, pos_wave, neg_wave), each shape [time].
    We'll pad them to [B, max_time].
    """
    anchors, positives, negatives = [], [], []
    for (a, p, n) in batch:
        anchors.append(a)
        positives.append(p)
        negatives.append(n)

    # shapes => [B, max_time]
    anchors = nn.utils.rnn.pad_sequence(anchors, batch_first=True)
    positives = nn.utils.rnn.pad_sequence(positives, batch_first=True)
    negatives = nn.utils.rnn.pad_sequence(negatives, batch_first=True)

    return anchors, positives, negatives

###############################################################################
# 4. Model: facebook/w2v-bert-2.0 => final projection w/ AutoFeatureExtractor
###############################################################################
class Wav2Vec2BertForTriplet(nn.Module):
    """
    - Loads AutoFeatureExtractor & AutoModel for "facebook/w2v-bert-2.0"
    - Takes waveforms => feature_extractor => base_model => final mean pooling => projection
    """
    def __init__(self, model_name="facebook/w2v-bert-2.0", embed_dim=256):
        super().__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name)  # Wav2Vec2BertModel
        hidden_size = self.base_model.config.hidden_size  # Typically 1024 from your config
        self.projection = nn.Linear(hidden_size, embed_dim)

    def forward(self, waveforms):
        # waveforms => [B, time] on device=mps (or cuda)
        
        # 1) Move waveforms to CPU for feature extraction
        waveforms_cpu = waveforms.detach().cpu()
        
        # 2) Convert to a list of numpy arrays if needed
        #    The feature extractor often accepts a list of audio arrays
        #    or a single 1D array. For a batch we do:
        speech_list = [waveforms_cpu[i].numpy() for i in range(waveforms_cpu.size(0))]

        # 3) Now do feature extraction on CPU arrays
        inputs = self.feature_extractor(
            speech_list,         # list of NumPy arrays
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # 4) Move the inputs to the correct device
        device = waveforms.device  # mps
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        
        # 5) Forward pass on base_model
        outputs = self.base_model(**inputs)
        hidden_states = outputs.last_hidden_state  # [B, seq_len, hidden_size]

        # 6) Pool & project
        pooled = hidden_states.mean(dim=1)
        emb = self.projection(pooled)
        return emb

###############################################################################
# 5. Training w/ TripletMarginLoss
###############################################################################
def train_triplet_margin(
    database_path,
    model_name="facebook/w2v-bert-2.0",
    embed_dim=256,
    margin=1.0,
    epochs=1,
    batch_size=2,
    lr=1e-5
):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"[INFO] Using device: {device}")

    # dataset & dataloader
    dataset = InTheWildTripletDataset(database_path=database_path, sr=16000)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=triplet_collate_fn
    )

    # model
    model = Wav2Vec2BertForTriplet(model_name=model_name, embed_dim=embed_dim)
    model.to(device)

    # triplet loss & optimizer
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_idx, (anchor_wavs, pos_wavs, neg_wavs) in enumerate(dataloader):
            anchor_wavs = anchor_wavs.to(device)
            pos_wavs    = pos_wavs.to(device)
            neg_wavs    = neg_wavs.to(device)

            optimizer.zero_grad()

            # forward => embeddings
            anchor_emb = model(anchor_wavs)
            pos_emb    = model(pos_wavs)
            neg_emb    = model(neg_wavs)

            # compute loss
            loss = triplet_loss_fn(anchor_emb, pos_emb, neg_emb)
            loss.backward()
            optimizer.step()

            bs = anchor_wavs.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{batch_idx+1}/{len(dataloader)}], "
                      f"Loss: {loss.item():.4f}")

        avg_loss = total_loss / total_samples if total_samples else 0
        print(f"Epoch [{epoch+1}/{epochs}] - Avg Triplet Loss: {avg_loss:.4f}")

    print("[INFO] Training complete.")
    return model

###############################################################################
# 6. Example Usage
###############################################################################
if __name__ == "__main__":
    database_path = "release_in_the_wild"  # Path to your meta.csv & .wav files
    model = train_triplet_margin(
        database_path=database_path,
        model_name="facebook/w2v-bert-2.0",  # or the correct HF model name
        embed_dim=256,
        margin=1.0,
        epochs=1,
        batch_size=2,
        lr=1e-5
    )
    # save model
    torch.save(model.state_dict(), "w2v_bert_triplet_model.pth")
    print("Model saved as w2v_bert_triplet_model.pth")
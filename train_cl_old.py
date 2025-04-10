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
      - lines: key, speaker, label
        'bona-fide' => label=1
        else => label=0
    Returns:
      d_meta   = dict {full_wav_path -> 0 or 1}
      file_list = all wav paths (spoofs first, then bona-fide)
    """
    file = os.path.join(database_path, 'meta.csv')
    d_meta = {}
    data_list0 = []  # label=0 (fake)
    data_list1 = []  # label=1 (real)

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
# 2. Triplet Dataset WITH Labels
###############################################################################
class InTheWildTripletDataset(Dataset):
    """
    Always:
      - anchor = real   (label=1)
      - positive = real (label=1)
      - negative = fake (label=0)

    So we must return 6 items:
      (anchor_wave, anchor_label, pos_wave, pos_label, neg_wave, neg_label)
    """
    def __init__(self, database_path, sr=16000, transform=None):
        super().__init__()
        self.database_path = database_path
        self.sr = sr
        self.transform = transform

        # parse meta.csv => get dict & full file_list
        d_meta, file_list = gen_in_the_wild_list_new(database_path)

        # separate real/fake paths
        real_paths = []  # label=1
        fake_paths = []  # label=0
        for fpath in file_list:
            if d_meta[fpath] == 1:
                real_paths.append(fpath)
            else:
                fake_paths.append(fpath)

        # store them
        self.real_paths = real_paths
        self.fake_paths = fake_paths
        self.d_meta = d_meta  # if you want to do any lookups later

        # The dataset length = number of real audio files
        # because each anchor must be real
        self.num_real = len(self.real_paths)

    def __len__(self):
        # We have one anchor per real audio
        return self.num_real

    def __getitem__(self, idx):
        """
        Returns:
          anchor_wave, anchor_label=1,
          pos_wave, pos_label=1,
          neg_wave, neg_label=0
        """
        # 1) Anchor
        anchor_path = self.real_paths[idx]
        anchor_waveform = self._load_audio(anchor_path)
        anchor_label = 1

        # 2) Positive: pick a random real
        pos_path = random.choice(self.real_paths)
        pos_waveform = self._load_audio(pos_path)
        pos_label = 1  # also real

        # 3) Negative: pick a random fake
        neg_path = random.choice(self.fake_paths)
        neg_waveform = self._load_audio(neg_path)
        neg_label = 0  # fake

        # Now we return 6 items as needed by hybrid_collate_fn
        return (anchor_waveform, anchor_label,
                pos_waveform, pos_label,
                neg_waveform, neg_label)

    def _load_audio(self, wav_path):
        """
        Loads single-channel audio at self.sr using librosa => torch FloatTensor [time].
        """
        waveform_np, sr_ = librosa.load(wav_path, sr=8000, mono=True)
        waveform_torch = torch.from_numpy(waveform_np)

        if self.transform:
            waveform_torch = self.transform(waveform_torch)

        return waveform_torch

###############################################################################
# 3. Collate function => pad waveforms, gather labels
###############################################################################
def hybrid_collate_fn(batch):
    """
    Expects each item in batch to be:
      (a_wav, a_lbl, p_wav, p_lbl, n_wav, n_lbl)

    We'll pad waveforms and then stack labels.
    Returns:
      anchor_wavs:  [B, max_time]
      anchor_labels:[B]
      pos_wavs:     [B, max_time]
      pos_labels:   [B]
      neg_wavs:     [B, max_time]
      neg_labels:   [B]
    """
    anchor_wavs, anchor_labels = [], []
    pos_wavs, pos_labels = [], []
    neg_wavs, neg_labels = [], []

    for item in batch:
        (a_wav, a_lbl, p_wav, p_lbl, n_wav, n_lbl) = item

        anchor_wavs.append(a_wav)
        anchor_labels.append(a_lbl)
        pos_wavs.append(p_wav)
        pos_labels.append(p_lbl)
        neg_wavs.append(n_wav)
        neg_labels.append(n_lbl)

    # pad waveforms to [B, max_time]
    anchor_wavs = nn.utils.rnn.pad_sequence(anchor_wavs, batch_first=True)
    pos_wavs    = nn.utils.rnn.pad_sequence(pos_wavs,    batch_first=True)
    neg_wavs    = nn.utils.rnn.pad_sequence(neg_wavs,    batch_first=True)

    # convert labels to LongTensors
    anchor_labels = torch.tensor(anchor_labels, dtype=torch.long)
    pos_labels    = torch.tensor(pos_labels,    dtype=torch.long)
    neg_labels    = torch.tensor(neg_labels,    dtype=torch.long)

    return (anchor_wavs, anchor_labels,
            pos_wavs, pos_labels,
            neg_wavs, neg_labels)

###############################################################################
# 4. Wav2Vec2BertHybrid Model => embeddings only
###############################################################################
class Wav2Vec2BertHybrid(nn.Module):
    """
    We'll produce embeddings for triplet margin.
    """
    def __init__(self, model_name="facebook/w2v-bert-2.0"):
        super().__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name)  # Wav2Vec2BertModel

        hidden_size = self.base_model.config.hidden_size  # typically 1024
       

    def forward(self, waveforms):
        """
        waveforms: [B, time]

        Returns:
          emb:   [B, embed_dim]
        """
        device = waveforms.device

        # Move waveforms to CPU for feature extraction
        waveforms_cpu = waveforms.detach().cpu()
        speech_list = [waveforms_cpu[i].numpy() for i in range(waveforms_cpu.size(0))]

        # Extract features on CPU
        inputs = self.feature_extractor(
            speech_list,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        # Move them back to the original device
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        outputs = self.base_model(**inputs)  # => [B, seq_len, hidden_size]
        hidden_states = outputs.last_hidden_state

        # mean pool
        pooled = hidden_states.mean(dim=1)  # => [B, hidden_size]

        return pooled

###############################################################################
# 5. Training with Triplet Margin Loss
###############################################################################
def train_triplet_margin(
    database_path,
    model_name="facebook/w2v-bert-2.0",
    margin=1.0,
    epochs=1,
    batch_size=2,
    lr=1e-5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # dataset & dataloader
    dataset = InTheWildTripletDataset(database_path=database_path, sr=16000)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=hybrid_collate_fn
    )

    # model
    model = Wav2Vec2BertHybrid(model_name=model_name)
    model.to(device)

    # loss
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # training
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_idx, batch in enumerate(dataloader):
            (anchor_wavs, anchor_labels,
             pos_wavs, pos_labels,
             neg_wavs, neg_labels) = batch

            # move waveforms to device
            anchor_wavs = anchor_wavs.to(device)
            pos_wavs = pos_wavs.to(device)
            neg_wavs = neg_wavs.to(device)

            optimizer.zero_grad()

            # 1) anchor => emb
            anchor_emb = model(anchor_wavs)
            # 2) pos => emb
            pos_emb = model(pos_wavs)
            # 3) neg => emb
            neg_emb = model(neg_wavs)

            # triplet margin loss
            loss_triplet = triplet_loss_fn(anchor_emb, pos_emb, neg_emb)

            loss_triplet.backward()
            optimizer.step()

            bs = anchor_wavs.size(0)
            total_loss += loss_triplet.item() * bs
            total_samples += bs

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{batch_idx+1}/{len(dataloader)}], "
                      f"Loss: {loss_triplet.item():.4f}")

        avg_loss = total_loss / total_samples if total_samples else 0
        print(f"Epoch [{epoch+1}/{epochs}] - Avg Loss: {avg_loss:.4f}")

    print("[INFO] Triplet margin training complete.")

    return model

###############################################################################
# 6. Example Usage
###############################################################################
if __name__ == "__main__":
    database_path = "../data/in_the_wild"  # Path to meta.csv & .wav files

    # We'll do 1 epoch just as a quick demonstration; adjust as desired
    model = train_triplet_margin(
        database_path=database_path,
        model_name="facebook/w2v-bert-2.0",
        margin=1.0,
        epochs=1,
        batch_size=4,
        lr=1e-5
    )

    # save model
    torch.save(model.state_dict(), "w2v_bert_hybrid_model_itw_triplet_loss.pth")
    print("Model saved as w2v_bert_hybrid_model_itw_triplet_loss.pth")

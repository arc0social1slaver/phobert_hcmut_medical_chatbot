import pandas as pd
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
# from torchmetrics.functional.classification import auroc
# from torch.amp.autocast_mode import autocast
# from torch.amp.grad_scaler import GradScaler
import torch.nn as nn
import torch
import torch.nn.functional as F
# from tqdm import tqdm
# from sklearn.metrics import f1_score, classification_report
# from pytorch_lamb import Lamb
from pytorch_lightning import LightningModule, Trainer, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

torch.set_float32_matmul_precision('medium')
MAX_SEQ_LEN = 223 + 2 # Old value: 195. Decide to keep as original
def read_file_json(fileName):
    with open(fileName, 'r', encoding='utf-8') as f:
        return json.load(f)
data = read_file_json("vimq_data/char2index.json")
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
batch_size = 32 # Will run value: 67
vocab = []
for char, _ in data.items():
    vocab.append(char)
# a b c d => 0 234 5555 5555 2 1 1 1 1 1 1 1
def get_token(tokenizer, words, max_seq_len = MAX_SEQ_LEN):
    inputs_id = [tokenizer.cls_token_id]
    for word in words:
        word_token = tokenizer.encode(word)
        inputs_id += word_token[1: (len(word_token) - 1)]
    inputs_id.append(tokenizer.sep_token_id)
    attention_mask = [1] * len(inputs_id)
    if len(inputs_id) > max_seq_len:
        inputs_id = inputs_id[:max_seq_len]
        attention_mask = attention_mask[:max_seq_len]
    else:
        inputs_id += [tokenizer.pad_token_id] * (max_seq_len - len(inputs_id))
        attention_mask += [0] * (max_seq_len - len(attention_mask))
    return inputs_id, attention_mask
def load_tokenizer():
    return AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
def clean_text(text: str):
    text = str(text).lower()
    cleaned_text = ''
    for char in text:
        if char in vocab:
            cleaned_text += char
    return cleaned_text
def cleanify_sentence(sentence):
    cleaned_text = ' '.join([clean_text(text) for text in sentence.split(" ")])
    # cleaned_text = cleaned_text.rsplit(".", 1)[0]
    return cleaned_text
data = pd.read_csv("vim-med/ViMedical_Disease.csv")
data["Question"] = data["Question"].apply(cleanify_sentence)
all_labels = data["Disease"].drop_duplicates().to_list()
labels_to_id = {label : i for i, label in enumerate(all_labels)}
id_to_labels = {i : label for i, label in enumerate(all_labels)}
data["Disease"] = data["Disease"].map(labels_to_id)
# split 80% 20% for validation
IDS_ALL_UNIQUE = data.shape[0]
np.random.seed(42)
train_idx = np.random.choice(np.arange(IDS_ALL_UNIQUE), int(0.8 * IDS_ALL_UNIQUE), replace=False)
valid_idx = np.setdiff1d(np.arange(IDS_ALL_UNIQUE), train_idx)
np.random.seed(None)

df_train = data.loc[train_idx].reset_index(drop=True)
df_val = data.loc[valid_idx].reset_index(drop=True)

tokenizer = load_tokenizer()
def tokenize_data(df, tokenizer_tool):
    list_input_ids = []
    list_attention_mask = []
    
    for text in df["Question"]:
        inputs_id, attention_mask = get_token(tokenizer_tool, text)
        list_input_ids.append(inputs_id)
        list_attention_mask.append(attention_mask)
    df["input_ids"] = list_input_ids
    df["attention_masks"] = list_attention_mask
    return df
df_train_tokenized = tokenize_data(df_train, tokenizer)
df_val_tokenized = tokenize_data(df_val, tokenizer)
class SentenceDataSet(Dataset):
    def __init__(self, df):
        self.input_ids = df['input_ids'].tolist()
        self.attention_masks = df['attention_masks'].tolist()
        self.labels = df['Disease'].tolist()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx]), torch.tensor(self.attention_masks[idx]), torch.tensor(self.labels[idx])
class QuickDataModule(LightningDataModule):
    def __init__(self, df_train = df_train_tokenized, df_val = df_val_tokenized, batch_size = batch_size, num_worker = 2):
        super().__init__()
        self.df_train = df_train
        self.df_val = df_val
        self.batch_size = batch_size
        self.num_worker = num_worker
    def setup(self, stage = None):
        if stage in (None, "fit"):
            self.train_dataset = SentenceDataSet(self.df_train)
            self.val_dataset = SentenceDataSet(self.df_val)
        if stage == 'predict':
            self.val_dataset = SentenceDataSet(self.df_val)
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_worker)
    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_worker)
    def predict_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_worker)
# Version 2 - The BiLSTM
class QuickModelLSTM(LightningModule):
    def __init__(self, num_labels, tokenizer_tool,  hidden_dim, lstm_layer, num_epoch, train_size = df_train.shape[0], batch_size = batch_size):
        super(QuickModelLSTM, self).__init__()
        # self.num_labels = num_labels
        self.save_hyperparameters()
        self.roberta = AutoModel.from_pretrained("vinai/phobert-base-v2", return_dict=True)
        
        self.lstm = nn.LSTM(self.roberta.config.hidden_size, self.hparams.hidden_dim, self.hparams.lstm_layer, batch_first=True, bidirectional=True, dropout=0.5)
        self.classfication = nn.Linear(self.hparams.hidden_dim * 2, self.hparams.num_labels)
        nn.init.xavier_uniform_(self.classfication.weight)
        self.loss_fnc = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(0.5)
        self.train_metrics = MetricCollection(
            {
                "accuracy": MulticlassAccuracy(num_classes=self.hparams.num_labels, average='weighted'),
                "f1": MulticlassF1Score(num_classes=self.hparams.num_labels, average='weighted'),
            },
            prefix="train_",
        )
        self.valid_metrics = self.train_metrics.clone(prefix="valid_")
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.roberta(input_ids = input_ids, attention_mask = attention_mask)
        sequence_output = output.last_hidden_state
        output, _ = self.lstm(sequence_output)
        pooled_output = torch.mean(output, 1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classfication(pooled_output)
        loss = 0
        if labels is not None:
            loss = self.loss_fnc(logits.view(-1, self.hparams.num_labels), labels.view(-1))
        return loss, logits
    def training_step(self, batch, batch_idx):
        id, mask, label = batch
        loss, logits = self(id, mask, label)
        self.log('train_loss', loss, prog_bar=True, on_step=True)
        batch_value = self.train_metrics(logits, label)
        self.log_dict(batch_value, prog_bar=True, on_step=True)
        return {"loss": loss, "predictions": logits, "labels": label}
    def validation_step(self, batch, batch_idx):
        id, mask, label = batch
        loss, logits = self(id, mask, label)
        self.log('val_loss', loss, prog_bar=True)
        self.valid_metrics.update(logits, label)
        return {"loss": loss, "predictions": logits, "labels": label}
    def predict_step(self, batch, batch_idx):
        id, mask, label = batch
        _, logits = self(id, mask, label)
        return logits
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1.5e-6, weight_decay=0.001)
        # optimizer = torch.optim.AdamW(self.parameters(), lr=1e-2, weight_decay=0.01)
        total_step = int(self.hparams.train_size / self.hparams.batch_size) * self.hparams.num_epoch
        warmup_step = int(total_step * 0.1)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_step, total_step)
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
    def on_validation_epoch_end(self):
        self.log_dict(self.valid_metrics.compute(), prog_bar=True)
        self.valid_metrics.reset()
    def on_train_epoch_end(self):
        self.train_metrics.reset()
modelSaver = ModelCheckpoint(monitor='val_loss',dirpath="work_progress/", filename='BiLSTM-Epoch-{epoch}-Val_loss-{val_loss:.4f}', save_top_k=1)
if __name__ == "__main__":
    epochs = 3
    data_module = QuickDataModule()
    data_module.setup()
    trainer = Trainer(max_epochs=epochs, num_sanity_val_steps=50, default_root_dir="my_board/", fast_dev_run=True, callbacks=[modelSaver])
    model = QuickModelLSTM(len(all_labels), tokenizer, 128, 2, epochs)
    trainer.fit(model, datamodule=data_module)
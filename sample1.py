import pandas as pd
import json
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from pyvi import ViTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
import re

torch.set_float32_matmul_precision('medium')
MAX_SEQ_LEN = 223 + 2
MAX_CHAR_LEN = 18
NUM_LAYER_PHOBERT = 3
def read_file_json(fileName):
    with open(fileName, 'r', encoding='utf-8') as f:
        return json.load(f)
vocab_to_id = read_file_json("vimq_data/char2index.json")
batch_size = 16 # Will run value: 67
vocabs = list(vocab_to_id.keys())
def get_token(tokenizer, words, max_seq_len = MAX_SEQ_LEN):
    inputs_id = [tokenizer.cls_token_id]
    all_index = [len(inputs_id)]
    for word in words:
        word_token = tokenizer.encode(word)
        inputs_id += word_token[1:-1]
        all_index.append(len(inputs_id))
    all_index = all_index[:-1]
    inputs_id.append(tokenizer.sep_token_id)
    attention_mask = [1] * len(inputs_id)
    if len(inputs_id) > max_seq_len:
        inputs_id = inputs_id[:max_seq_len]
        attention_mask = attention_mask[:max_seq_len]
        all_index = all_index[:max_seq_len]
    else:
        inputs_id += [tokenizer.pad_token_id] * (max_seq_len - len(inputs_id))
        attention_mask += [0] * (max_seq_len - len(attention_mask))
        all_index += [0] * (max_seq_len - len(all_index))
    return inputs_id, attention_mask, all_index
def get_chars(words, max_char_len = MAX_CHAR_LEN):
    lst_char = []
    for word in words:
        lst_item = []
        for i in range(max_char_len):
            try:
                char = word[i]
            except:
                char = "PAD"
            lst_item.append(char)
        lst_char.append(lst_item)
    return lst_char
def encode_chars(char_set, max_seq_len = MAX_SEQ_LEN):
    lst_char = []
    for char in char_set:
        lst_item = []
        for item in char:
            if item not in vocabs:
                lst_item.append(vocab_to_id.get("UNK"))
            else:
                lst_item.append(vocab_to_id.get(item))
        lst_char.append(lst_item)
    if len(lst_char) < max_seq_len:
        padding_word = [[vocab_to_id.get("PAD")]* MAX_CHAR_LEN]
        lst_char += padding_word * (max_seq_len - len(lst_char))
    else:
        lst_char = lst_char[:max_seq_len]
    return lst_char
def load_tokenizer():
    return AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
def cleanify_sentence(sentence : str):
    cleaned_text = sentence.lower()
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return ViTokenizer.tokenize(cleaned_text).split(' ')
data = pd.read_csv("vim-med/ViMedical_Disease.csv")
data["Question"] = data["Question"].apply(cleanify_sentence)
all_labels = data["Disease"].drop_duplicates().to_list()
labels_to_id = {label : i for i, label in enumerate(all_labels)}
id_to_labels = {i : label for i, label in enumerate(all_labels)}
data["Disease"] = data["Disease"].map(labels_to_id)

df_train, df_val = train_test_split(data, test_size=0.2, stratify=data["Disease"], random_state=42)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
tokenizer = load_tokenizer()
def tokenize_data(df, tokenizer_tool):
    list_input_ids = []
    list_attention_mask = []
    list_index = []
    list_charid = []
    
    for text in df["Question"]:
        inputs_id, attention_mask, all_indexes = get_token(tokenizer_tool, text)
        char_set = get_chars(text)
        encode_char_set = encode_chars(char_set)
        list_input_ids.append(inputs_id)
        list_attention_mask.append(attention_mask)
        list_index.append(all_indexes)
        list_charid.append(encode_char_set)
    df["input_ids"] = list_input_ids
    df["attention_masks"] = list_attention_mask
    df["all_index"] = list_index
    df["char_id"] = list_charid
    return df
df_train_tokenized = tokenize_data(df_train, tokenizer)
df_val_tokenized = tokenize_data(df_val, tokenizer)
class SentenceDataSet(Dataset):
    def __init__(self, df):
        self.input_ids = df['input_ids'].tolist()
        self.attention_masks = df['attention_masks'].tolist()
        self.all_index = df["all_index"].tolist()
        self.char_id = df["char_id"].tolist()
        self.labels = df["Disease"].tolist()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx]), torch.tensor(self.attention_masks[idx]), torch.tensor(self.all_index[idx]), torch.tensor(self.char_id[idx]) , torch.tensor(self.labels[idx])
class QuickDataModule(LightningDataModule):
    def __init__(self, df_train = df_train_tokenized, df_val = df_val_tokenized, batch_size = batch_size, num_worker = 4):
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


# # Version 2 - The BiLSTM
class QuickModelLSTM(LightningModule):
    def __init__(self, num_labels, tokenizer_tool,  hidden_dim, lstm_layer, num_epoch, train_size = df_train.shape[0], batch_size = batch_size):
        super(QuickModelLSTM, self).__init__()
        self.save_hyperparameters()
        self.roberta = AutoModel.from_pretrained("vinai/phobert-base-v2")
        
        self.wordEmbeds = nn.Embedding(108, 100)

        self.conv1 = nn.Sequential(
            nn.Conv1d(100, 128, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(100, 128, kernel_size=4, padding=1),
            nn.Tanh(),
        )
        self.dropoutChar = nn.Dropout(0.3)

        self.lstm = nn.LSTM(self.roberta.config.hidden_size * NUM_LAYER_PHOBERT + 2 * 128, self.hparams.hidden_dim, self.hparams.lstm_layer, batch_first=True, bidirectional=True, dropout=0.5)
        self.classfication = nn.Linear(self.hparams.hidden_dim * 2, self.hparams.num_labels)
        nn.init.xavier_uniform_(self.classfication.weight)
        nn.init.xavier_uniform_(self.wordEmbeds.weight)
        self.loss_fnc = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout()
        self.dropout1 = nn.Dropout(0.3)
        self.train_metrics = MetricCollection(
            {
                "accuracy": MulticlassAccuracy(num_classes=self.hparams.num_labels, average='weighted'),
                "f1": MulticlassF1Score(num_classes=self.hparams.num_labels, average='weighted'),
            },
            prefix="train_",
        )
        self.valid_metrics = self.train_metrics.clone(prefix="valid_")
    def forward(self, input_ids, attention_mask, all_index, char_id, labels=None):
        outputs = self.roberta(input_ids = input_ids, attention_mask = attention_mask, output_hidden_states = True)
        bert_output = []
        for i in range(-1, -(NUM_LAYER_PHOBERT + 1), -1):
            one_layer = outputs.hidden_states[i]
            bert_output.append(one_layer)
        bert_output = torch.cat(bert_output, dim=-1)
        bert_output = torch.cat([
            torch.index_select(bert_output[i], 0, all_index[i]).unsqueeze(0)
            for i in range(bert_output.size(0))
        ], dim=0)
        bert_output = self.dropout1(bert_output)
        ## char Embeding
        bs = char_id.size(0)
        seq_len = char_id.size(1)
        char_output = char_id.view(bs * seq_len, char_id.size(2))
        char_output = self.wordEmbeds(char_output)
        char_output = char_output.transpose(2, 1)
        
        cnn_output = [self.conv1(char_output), self.conv2(char_output)]
        cnn_output = [F.max_pool1d(op, op.size(2)) for op in cnn_output]

        char_output = torch.cat(cnn_output, 1)
        char_output = char_output.view(bs, seq_len, -1)
        char_output = self.dropoutChar(char_output)


        sequence_output = torch.cat((bert_output, char_output), dim=-1)

        outputs, _ = self.lstm(sequence_output)
        pooled_output = torch.mean(outputs, 1)
        pooled_output = self.dropout(pooled_output)

        logits = self.classfication(pooled_output)
        loss = 0
        if labels is not None:
            loss = self.loss_fnc(logits.view(-1, self.hparams.num_labels), labels.view(-1))
        return loss, logits
    def training_step(self, batch, batch_idx):
        input_id_batch, mask, all_index, char_id, labels_batch = batch
        loss, logits = self(input_id_batch, mask, all_index, char_id, labels_batch)
        self.log('train_loss', loss, prog_bar=True, on_step=True)
        batch_value = self.train_metrics(logits, labels_batch)
        self.log_dict(batch_value)
        return {"loss": loss, "predictions": logits, "labels": labels_batch}
    def validation_step(self, batch, batch_idx):
        input_id_batch, mask, all_index, char_id, labels_batch = batch
        loss, logits = self(input_id_batch, mask, all_index, char_id, labels_batch)
        self.log('val_loss', loss, prog_bar=True)
        self.valid_metrics.update(logits, labels_batch)
        return {"loss": loss, "predictions": logits, "labels": labels_batch}
    def predict_step(self, batch, batch_idx):
        input_id_batch, mask, all_index, char_id, labels_batch = batch
        _, logits = self(input_id_batch, mask, all_index, char_id, labels_batch)
        return logits
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-5)

        total_step = int(self.hparams.train_size / self.hparams.batch_size) * self.hparams.num_epoch
        warmup_step = int(total_step * 0.1)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_step, total_step)
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
    def on_validation_epoch_end(self):
        self.log_dict(self.valid_metrics.compute(), prog_bar=True)
        self.valid_metrics.reset()
    def on_train_epoch_end(self):
        self.train_metrics.reset()


# Version 1
class QuickModel(LightningModule):
    def __init__(self, num_labels, tokenizer_tool, num_epoch, train_size = df_train.shape[0], batch_size = batch_size):
        super(QuickModel, self).__init__()
        # self.num_labels = num_labels
        self.save_hyperparameters()
        self.roberta = AutoModel.from_pretrained("vinai/phobert-base-v2")
        
        self.wordEmbeds = nn.Embedding(108, 100)

        self.conv1 = nn.Sequential(
            nn.Conv1d(100, 128, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(100, 128, kernel_size=4, padding=1),
            nn.Tanh(),
        )
        self.dropoutChar = nn.Dropout(0.3)

        self.hidden = nn.Linear(self.roberta.config.hidden_size * NUM_LAYER_PHOBERT + 2 * 128, self.roberta.config.hidden_size)

        self.classfication = nn.Linear(self.roberta.config.hidden_size, self.hparams.num_labels)
        nn.init.xavier_uniform_(self.wordEmbeds.weight)
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.xavier_uniform_(self.classfication.weight)

        self.loss_fnc = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout()
        self.dropout1 = nn.Dropout(0.3)
        self.train_metrics = MetricCollection(
            {
                "accuracy": MulticlassAccuracy(num_classes=self.hparams.num_labels, average='weighted'),
                "f1": MulticlassF1Score(num_classes=self.hparams.num_labels, average='weighted'),
            },
            prefix="train_",
        )
        self.valid_metrics = self.train_metrics.clone(prefix="valid_")
    def forward(self, input_ids, attention_mask, all_index, char_id, labels=None):
        output = self.roberta(input_ids = input_ids, attention_mask = attention_mask, output_hidden_states = True)
        bert_output = []
        for i in range(-1, -(NUM_LAYER_PHOBERT + 1), -1):
            one_layer = output.hidden_states[i]
            bert_output.append(one_layer)
        bert_output = torch.cat(bert_output, dim=-1)
        bert_output = torch.cat([
            torch.index_select(bert_output[i], 0, all_index[i]).unsqueeze(0)
            for i in range(bert_output.size(0))
        ], dim=0)
        bert_output = self.dropout1(bert_output)
        ## char Embeding
        bs = char_id.size(0)
        seq_len = char_id.size(1)
        char_output = char_id.view(bs * seq_len, char_id.size(2))
        char_output = self.wordEmbeds(char_output)
        char_output = char_output.transpose(2, 1)
        
        cnn_output = [self.conv1(char_output), self.conv2(char_output)]
        cnn_output = [F.max_pool1d(op, op.size(2)) for op in cnn_output]

        char_output = torch.cat(cnn_output, 1)
        char_output = char_output.view(bs, seq_len, -1)
        char_output = self.dropoutChar(char_output)



        sequence_output = torch.cat((bert_output, char_output), dim=-1)

        pooled_output = torch.mean(sequence_output, 1)

        
        pooled_output = self.hidden(pooled_output)
        pooled_output = self.dropout(pooled_output)
        pooled_output = F.relu(pooled_output)
        
        
        logits = self.classfication(pooled_output)
        loss = 0
        if labels is not None:
            loss = self.loss_fnc(logits.view(-1, self.hparams.num_labels), labels.view(-1))
        return loss, logits
    def training_step(self, batch, batch_idx):
        input_id_batch, mask, all_index, char_id, true_labels = batch
        loss, logits = self(input_id_batch, mask, all_index, char_id, true_labels)
        self.log('train_loss', loss, prog_bar=True, on_step=True)
        batch_value = self.train_metrics(logits, true_labels)
        self.log_dict(batch_value)
        return {"loss": loss, "predictions": logits, "labels": true_labels}
    def validation_step(self, batch, batch_idx):
        input_id_batch, mask, all_index, char_id, true_labels = batch
        loss, logits = self(input_id_batch, mask, all_index, char_id, true_labels)
        self.log('val_loss', loss, prog_bar=True)
        self.valid_metrics.update(logits, true_labels)
        return {"loss": loss, "predictions": logits, "labels": true_labels}
    def predict_step(self, batch, batch_idx):
        input_id_batch, mask, all_index, char_id, true_labels = batch
        _, logits = self(input_id_batch, mask, all_index, char_id, true_labels)
        return logits
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-5)

        total_step = int(self.hparams.train_size / self.hparams.batch_size) * self.hparams.num_epoch
        warmup_step = int(total_step * 0.1)

        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_step, total_step)
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
    def on_validation_epoch_end(self):
        self.log_dict(self.valid_metrics.compute(), prog_bar=True)
        self.valid_metrics.reset()
    def on_train_epoch_end(self):
        self.train_metrics.reset()


modelSaver = ModelCheckpoint(monitor='val_loss',dirpath="work_progress/", filename='BestModelLoss-{epoch}-{val_loss:.4f}', save_top_k=1)
modelAcc = ModelCheckpoint(monitor='valid_accuracy',dirpath="work_progress/", mode='max', filename='BestModelAccuracy-{epoch}-{valid_accuracy:.4f}', save_top_k=1)
modelF1 = ModelCheckpoint(monitor='valid_f1',dirpath="work_progress/",mode='max', filename='BestModelF1-{epoch}-{valid_f1:.4f}', save_top_k=1)
epochs = 12
data_module = QuickDataModule()
data_module.setup()

trainer = Trainer(max_epochs=epochs, num_sanity_val_steps=50, default_root_dir="my_board/", fast_dev_run=False, callbacks=[modelSaver, modelAcc, modelF1], gradient_clip_val=1.0, gradient_clip_algorithm="norm")

# model = QuickModelLSTM(len(all_labels), tokenizer, 512, 2, epochs)
model = QuickModel(len(all_labels), tokenizer, epochs)
trainer.fit(model, datamodule=data_module)
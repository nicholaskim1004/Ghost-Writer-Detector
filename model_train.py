#Script to run specified models
import os
import random
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#model class
class MultiClassRegression(nn.Module):
    def __init__(self, dim, num_classes):
        super(MultiClassRegression, self).__init__()
        self.layer1 = nn.Linear(dim, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        probs = self.layer1(x)
        probs = self.relu(probs)
        probs = self.layer2(probs)
        return probs
    
class MultiClassRegressionBig(nn.Module):
    def __init__(self, dim, num_classes):
        super(MultiClassRegressionBig, self).__init__()
        self.layer1 = nn.Linear(dim, 1000)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(1000, 128)
        self.layer3 = nn.Linear(128,32)
        self.layer_out = nn.Linear(32,num_classes)
        
    def forward(self, x):
        probs = self.layer1(x)
        probs = self.relu(probs)
        probs = self.layer2(probs)
        probs = self.relu(probs)
        probs = self.layer3(probs)
        probs = self.relu(probs)
        probs = self.layer_out(probs)
        
        return probs

#helper function to convert data for bert
def dataset_to_bert(texts, labels, tokenizer):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )
    return torch.utils.data.TensorDataset(
        encoded['input_ids'],
        encoded['attention_mask'],
        torch.tensor(labels, dtype=torch.long)
    )

class BertTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        t = self.texts[idx]
        enc = self.tokenizer(
            t,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        # squeeze to remove batch dim
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_ids, attention_mask, label


# ---------------- Base Trainer ----------------
class BaseTrainer:
    def __init__(
        self,
        data,
        text_col: str = "main_artist_lyrics_joined",
        artist_col: str = "artist",
        test_mask_col: Optional[str] = "pot_ghost",
        test_mask_value: int = 1,
        val_size: float = 0.2,
        random_state: int = 42,
        output_dir: str = "models",
        device: Optional[torch.device] = None,
    ):
        """
        data: pandas DataFrame containing at least text_col and artist_col.
        test_mask_col: if provided, rows where data[test_mask_col] == test_mask_value
                       will be used as test set (mirrors your original code).
        """
        self.raw_data = data.copy()
        self.text_col = text_col
        self.artist_col = artist_col
        self.test_mask_col = test_mask_col
        self.test_mask_value = test_mask_value
        self.val_size = val_size
        self.random_state = random_state
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Build artist mapping and basic splits
        self._build_artist_mapping()
        self._prepare_splits()

    def _build_artist_mapping(self):
        all_artists = self.raw_data[self.artist_col].unique().tolist()
        self.artist_mapping: Dict[str, int] = {artist: idx for idx, artist in enumerate(all_artists)}
        self.num_classes = len(self.artist_mapping)
        print(f"[BaseTrainer] Found {self.num_classes} artists / classes.")

    def _prepare_splits(self):
        # mirror your original logic: test_data is rows where pot_ghost == 1
        if self.test_mask_col is not None and self.test_mask_col in self.raw_data.columns:
            test_data = self.raw_data[self.raw_data[self.test_mask_col] == self.test_mask_value].copy()
            train_val_data = self.raw_data[self.raw_data[self.test_mask_col] != self.test_mask_value].copy()
        else:
            # if no test mask provided, do a simple train/test split
            test_data = self.raw_data.sample(frac=0.1, random_state=self.random_state)
            train_val_data = self.raw_data.drop(test_data.index)

        # map artist ids
        train_val_data["artist_id"] = train_val_data[self.artist_col].map(self.artist_mapping)
        test_data["artist_id"] = test_data[self.artist_col].map(self.artist_mapping)

        # keep only the columns we need
        self.test_data = test_data[[self.text_col, "artist_id"]].reset_index(drop=True)
        train_val_df = train_val_data[[self.text_col, "artist_id"]].reset_index(drop=True)

        # stratified train/val split
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=train_val_df["artist_id"],
        )

        self.train_data = train_df.reset_index(drop=True)
        self.val_data = val_df.reset_index(drop=True)

        print(f"[BaseTrainer] train: {len(self.train_data)}, val: {len(self.val_data)}, test: {len(self.test_data)}")

    def save_model(self, model: nn.Module, model_type: str):
        model_state_dict = model.state_dict()
        PATH = os.path.join(self.output_dir, f"model_parameters_{model_type}.pth")
        torch.save(model_state_dict, PATH)
        print(f"[BaseTrainer] Saved model parameters to {PATH}")


# ---------------- BERT Trainer ----------------
class BertTrainer(BaseTrainer):
    def __init__(
        self,
        *args,
        freeze_bert: bool = True,  # support both frozen and fine-tune
        bert_model_name: str = "bert-base-uncased",
        batch_size: int = 16,
        lr: float = 1e-4,
        num_epochs: int = 3,
        max_length: int = 128,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.freeze_bert = freeze_bert
        self.bert_model_name = bert_model_name
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.max_length = max_length

        # prepare tokenizer and model
        print("[BertTrainer] Loading tokenizer and model...")
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
        self.bert = BertModel.from_pretrained(self.bert_model_name)

        # classifier: user used MultiClassRegression(bert_hidden_size, num_classes)
        bert_hidden_size = self.bert.config.hidden_size
        # instantiate classifier (assumes MultiClassRegression exists)
        self.classifier = MultiClassRegression(bert_hidden_size, self.num_classes).to(self.device)

        # send bert to device too (even if frozen we may want it on same device)
        self.bert.to(self.device)

        if self.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            print("[BertTrainer] BERT params frozen (only classifier will train).")
        else:
            for param in self.bert.parameters():
                param.requires_grad = True
            print("[BertTrainer] BERT params will be fine-tuned.")

        # prepare datasets / loaders
        self._prepare_dataloaders()

        # choose optimizer:
        # if bert is trainable, include its params; else only classifier params
        if self.freeze_bert:
            self.optimizer = AdamW(self.classifier.parameters(), lr=self.lr)
        else:
            params = list(self.bert.parameters()) + list(self.classifier.parameters())
            self.optimizer = AdamW(params, lr=self.lr)

        self.loss_fn = nn.CrossEntropyLoss()

    def _prepare_dataloaders(self):
        train_texts = self.train_data[self.text_col].tolist()
        train_labels = self.train_data["artist_id"].tolist()
        val_texts = self.val_data[self.text_col].tolist()
        val_labels = self.val_data["artist_id"].tolist()

        train_ds = BertTextDataset(train_texts, train_labels, self.tokenizer, max_length=self.max_length)
        val_ds = BertTextDataset(val_texts, val_labels, self.tokenizer, max_length=self.max_length)

        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=self.batch_size)
        print(f"[BertTrainer] DataLoaders ready (train batches: {len(self.train_loader)}, val batches: {len(self.val_loader)})")

    def train(self):
        acc_list = []
        random.seed(1234)
        for epoch in range(self.num_epochs):
            self.classifier.train()
            if not self.freeze_bert:
                self.bert.train()
            else:
                self.bert.eval()

            for input_ids, attention_mask, labels in self.train_loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                # if frozen: we don't want gradients in BERT encoder
                if self.freeze_bert:
                    with torch.no_grad():
                        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                        embeddings = bert_out.pooler_output  # [batch_size, hidden]
                    outputs = self.classifier(embeddings)
                    loss = self.loss_fn(outputs, labels)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                else:
                    # fine-tune BERT + classifier in usual manner
                    bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                    embeddings = bert_out.pooler_output
                    outputs = self.classifier(embeddings)
                    loss = self.loss_fn(outputs, labels)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            # validation
            self.classifier.eval()
            self.bert.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for input_ids, attention_mask, labels in self.val_loader:
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    labels = labels.to(self.device)

                    bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                    embeddings = bert_out.pooler_output
                    outputs = self.classifier(embeddings)
                    preds = outputs.argmax(dim=1)

                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())

            acc = accuracy_score(all_labels, all_preds)
            acc_list.append(acc)
            print(f"[BertTrainer] Epoch {epoch+1}/{self.num_epochs}, Validation Accuracy: {acc:.4f}")

        # save classifier weights (and optionally you'd like to save bert too if fine-tuned)
        if self.freeze_bert:
            self.save_model(self.classifier, "bert_frozen")
        else:
            # save both: classifier + bert
            self.save_model(self.classifier, "bert_finetuned_classifier")
            bert_path = os.path.join(self.output_dir, "bert_model_finetuned.pth")
            torch.save(self.bert.state_dict(), bert_path)
            print(f"[BertTrainer] Saved bert state dict to {bert_path}")

        return acc_list


# ---------------- TF-IDF Full-batch Trainer ----------------
class TfIdfTrainer(BaseTrainer):
    def __init__(
        self,
        *args,
        max_features: int = 50000,
        ngram_range: tuple = (1, 2),
        lr: float = 1e-4,
        num_epochs: int = 200,
        convert_to_dense: bool = True,
        **kwargs,
    ):
        """
        convert_to_dense: if True, convert tf-idf output to dense numpy arrays before creating torch tensors.
                          Set to False only if you adapt your model to accept sparse inputs.
        """
        super().__init__(*args, **kwargs)
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.lr = lr
        self.num_epochs = num_epochs
        self.convert_to_dense = convert_to_dense

        self.tfidf = TfidfVectorizer(lowercase=True, max_features=self.max_features, ngram_range=self.ngram_range)
        self.loss_fn = nn.CrossEntropyLoss()

        self._prepare_data()
        # instantiate model (assumes MultiClassRegression exists)
        self.model = MultiClassRegression(self.X_train.shape[1], self.num_classes).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)

    def _prepare_data(self):
        # fit tfidf on train set
        X_train = self.tfidf.fit_transform(self.train_data[self.text_col])
        X_val = self.tfidf.transform(self.val_data[self.text_col])

        if self.convert_to_dense:
            X_train = X_train.toarray()
            X_val = X_val.toarray()
        else:
            raise RuntimeError("Sparse training not implemented by TfIdfTrainer. Set convert_to_dense=True or use a sparse-aware trainer.")

        # create torch tensors
        self.X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(self.train_data["artist_id"].values, dtype=torch.long).to(self.device)
        self.X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        self.y_val = torch.tensor(self.val_data["artist_id"].values, dtype=torch.long).to(self.device)

        print(f"[TfIdfTrainer] Data prepared (X_train shape: {self.X_train.shape})")

    def train(self):
        val_loss = []
        random.seed(123)
        print("[TfIdfTrainer] Starting training 🚀")
        for epoch in range(self.num_epochs):
            self.model.train()
            predicted = self.model(self.X_train)
            loss = self.loss_fn(predicted, self.y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 50 == 0 or epoch == self.num_epochs - 1:
                self.model.eval()
                with torch.no_grad():
                    predicted_val = self.model(self.X_val)
                    pred_label = torch.argmax(predicted_val, dim=1)
                    loss_val = self.loss_fn(predicted_val, self.y_val)
                    val_loss.append(loss_val.item())
                    val_acc = accuracy_score(self.y_val.cpu().numpy(), pred_label.cpu().numpy())
                    print(f"[TfIdfTrainer] epoch:{epoch+1}/{self.num_epochs}, loss:{loss.item():.4f}, val_loss:{loss_val.item():.4f}, val_acc:{val_acc:.4f}")

        # save
        self.save_model(self.model, "tfidf_fullbatch")
        return val_loss


# ---------------- TF-IDF SGD Trainer (per-sample updates) ----------------
class TfIdfSGDTrainer(BaseTrainer):
    def __init__(
        self,
        *args,
        max_features: int = 50000,
        ngram_range: tuple = (1, 2),
        lr: float = 1e-4,
        num_epochs: int = 200,
        big: bool = False,
        convert_to_dense: bool = True,
        **kwargs,
    ):
        """
        big: if True, instantiate MultiClassRegressionBig (your 'big' model), else MultiClassRegression.
        convert_to_dense: training uses dense arrays (row sampling uses numpy indexing)
        """
        super().__init__(*args, **kwargs)
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.lr = lr
        self.num_epochs = num_epochs
        self.big = big
        self.convert_to_dense = convert_to_dense

        self.tfidf = TfidfVectorizer(lowercase=True, max_features=self.max_features, ngram_range=self.ngram_range)
        self.loss_fn = nn.CrossEntropyLoss()

        self._prepare_data()

        # instantiate model
        if self.big:
            self.model = MultiClassRegressionBig(self.X_train_np.shape[1], self.num_classes).to(self.device)
        else:
            self.model = MultiClassRegression(self.X_train_np.shape[1], self.num_classes).to(self.device)

        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)

    def _prepare_data(self):
        X_train = self.tfidf.fit_transform(self.train_data[self.text_col])
        X_val = self.tfidf.transform(self.val_data[self.text_col])

        if self.convert_to_dense:
            X_train = X_train.toarray()
            X_val = X_val.toarray()
        else:
            raise RuntimeError("Sparse training not implemented in TfIdfSGDTrainer. Set convert_to_dense=True.")

        # store numpy arrays for easy random sampling per-index
        self.X_train_np = X_train  # numpy array
        self.y_train_np = self.train_data["artist_id"].values
        self.X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        self.y_val = torch.tensor(self.val_data["artist_id"].values, dtype=torch.long).to(self.device)

        print(f"[TfIdfSGDTrainer] Data prepared (X_train_np shape: {self.X_train_np.shape})")

    def train(self):
        val_loss = []
        random.seed(123)

        n = len(self.X_train_np)
        print("[TfIdfSGDTrainer] Starting SGD-style training 🚀")
        for epoch in range(self.num_epochs):
            self.model.train()
            # go through n updates (approx one epoch as in your original code)
            last_loss = None
            for _ in range(n):
                index = np.random.randint(0, n)
                x_row = torch.tensor(self.X_train_np[index]).unsqueeze(0).to(self.device)
                y_row = torch.tensor(self.y_train_np[index]).unsqueeze(0).to(self.device)

                predicted = self.model(x_row)
                loss = self.loss_fn(predicted, y_row)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                last_loss = loss.item()

            # Validation at end of epoch
            self.model.eval()
            with torch.no_grad():
                predicted_val = self.model(self.X_val)
                pred_label = torch.argmax(predicted_val, dim=1)
                loss_val = self.loss_fn(predicted_val, self.y_val)
                val_loss.append(loss_val.item())
                val_acc = accuracy_score(self.y_val.cpu().numpy(), pred_label.cpu().numpy())
                print(f"[TfIdfSGDTrainer] epoch:{epoch+1}/{self.num_epochs}, last_loss:{last_loss:.4f}, val_loss:{loss_val.item():.4f}, val_acc:{val_acc:.4f}")

        # save with different suffix depending on big
        suffix = "tfidf_sgd_big" if self.big else "tfidf_sgd"
        self.save_model(self.model, suffix)
        return val_loss


# ---------------- Trainer factory ----------------
def get_trainer(
    model_type: str,
    data,
    **kwargs,
):
    """
    model_type: "bert", "tfidf", "tfidf_sgd", or "tfidf_sgd_big"
    kwargs are passed to the trainer constructors (num_epochs, lr, freeze_bert, etc)
    """
    model_type = model_type.lower()
    if model_type == "bert":
        return BertTrainer(data, **kwargs)
    elif model_type == "tfidf":
        return TfIdfTrainer(data, **kwargs)
    elif model_type == "tfidf_sgd":
        return TfIdfSGDTrainer(data, big=False, **kwargs)
    elif model_type == "tfidf_sgd_big":
        return TfIdfSGDTrainer(data, big=True, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ---------------- Example usage ----------------
if __name__ == "__main__":
    import pandas as pd

    hip = pd.read_csv("data/cleaned_hip_dat.csv")
    try:
        hip  
    except NameError:
        print("Please provide a pandas DataFrame named `hip` or call get_trainer(...) with your dataframe.")
    else:
        
        sgd_trainer = get_trainer("tfidf_sgd_big", hip, num_epochs=15, lr=1e-4)
        sgd_trainer.train()
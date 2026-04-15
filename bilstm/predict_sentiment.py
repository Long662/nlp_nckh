import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support,
)
import matplotlib.pyplot as plt
import seaborn as sns

# ================== Config ==================
MODEL_PATH = "bilstm_vn_sentiment.pt"
TEST_PATH  = r"../data/test/test.csv"
BATCH_SIZE = 64
SEQ_LEN    = 64
N_BENCH    = 1000
TEST_TEXT  = "Sản phẩm dùng khá ổn, chất lượng tạm chấp nhận được."

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ================== Load checkpoint ==================
ckpt     = torch.load(MODEL_PATH, map_location=DEVICE)
word2idx = ckpt["word2idx"]
id2label = {int(k): v for k, v in ckpt["id2label"].items()}
label2id = ckpt["label2id"]
cfg      = ckpt["config"]
FIVE_CLASS = [id2label[i] for i in range(cfg["num_classes"])]

PAD_IDX = 0

# ================== Model (giống lúc train) ==================
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout=0.35):
        super().__init__()
        self.embedding   = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.spatial_drop = nn.Dropout(0.2)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 4, 192)
        self.fc2 = nn.Linear(192, num_classes)

    def forward(self, x):
        emb = self.spatial_drop(self.embedding(x))
        out, _ = self.bilstm(emb)
        avg_pool = out.mean(dim=1)
        max_pool = out.max(dim=1).values
        x = torch.cat([avg_pool, max_pool], dim=1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

model = BiLSTMClassifier(
    cfg["vocab_size"], cfg["embed_dim"],
    cfg["bilstm_units"], cfg["num_classes"],
).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Loaded model from {MODEL_PATH}")

# ================== Encode ==================
def encode(text):
    tokens = str(text).lower().split()[:SEQ_LEN]
    ids = [word2idx.get(t, 1) for t in tokens]
    ids += [PAD_IDX] * (SEQ_LEN - len(ids))
    return ids

# ================== Dataset ==================
class TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.X = [torch.tensor(encode(t), dtype=torch.long) for t in texts]
        self.y = labels

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        if self.y is not None:
            return self.X[i], self.y[i]
        return self.X[i]

# ================== Predict function ==================
def predict(texts):
    ds = TextDataset(texts)
    loader = DataLoader(ds, batch_size=BATCH_SIZE)
    all_probs = []
    with torch.no_grad():
        for xb in loader:
            logits = model(xb.to(DEVICE))
            probs  = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
    probs = np.vstack(all_probs)
    preds = probs.argmax(axis=1)
    return preds, probs

# ================== Evaluate test.csv ==================
df = pd.read_csv(TEST_PATH).dropna(subset=["text", "label"])
df["label"] = df["label"].astype(str).str.strip()
df = df[df["label"].isin(FIVE_CLASS)]

texts      = df["text"].astype(str).tolist()
true_ids   = [label2id[l] for l in df["label"]]
true_labels = df["label"].tolist()

preds, probs = predict(texts)
pred_labels  = [id2label[i] for i in preds]

print("\n===== BiLSTM — Kết quả trên test.csv =====")
print(classification_report(true_labels, pred_labels, digits=4, zero_division=0))

acc  = accuracy_score(true_labels, pred_labels)
prec, rec, f1, _ = precision_recall_fscore_support(
    true_labels, pred_labels, average="weighted", zero_division=0
)
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")

# ================== Confusion matrix ==================
cm = confusion_matrix(true_labels, pred_labels, labels=FIVE_CLASS)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=FIVE_CLASS, yticklabels=FIVE_CLASS, ax=axes[0])
axes[0].set_title("Confusion Matrix (count)")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
axes[0].set_xticklabels(FIVE_CLASS, rotation=45, ha="right")

row_sums = cm.sum(axis=1, keepdims=True)
cm_norm  = np.divide(cm.astype(float), row_sums, where=row_sums != 0)
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=FIVE_CLASS, yticklabels=FIVE_CLASS, ax=axes[1])
axes[1].set_title("Confusion Matrix (normalized)")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
axes[1].set_xticklabels(FIVE_CLASS, rotation=45, ha="right")

plt.tight_layout()
plt.savefig("confusion_matrix_bilstm.png", dpi=150)
plt.show()
print("Saved confusion_matrix_bilstm.png")

# ================== Benchmark tốc độ ==================
x_bench = torch.tensor([encode(TEST_TEXT)], dtype=torch.long).to(DEVICE)

with torch.no_grad():
    _ = model(x_bench)  # warm-up

t0 = time.time()
with torch.no_grad():
    for _ in range(N_BENCH):
        _ = model(x_bench)
t1 = time.time()

avg_ms = (t1 - t0) / N_BENCH * 1000
pred_label = id2label[int(model(x_bench).argmax(dim=1).item())]

print(f"\n===== Benchmark tốc độ (CPU/GPU) =====")
print(f"Text          : {TEST_TEXT}")
print(f"Prediction    : {pred_label}")
print(f"Số lần lặp    : {N_BENCH}")
print(f"Tổng thời gian: {t1-t0:.4f} s")
print(f"Thời gian/câu : {avg_ms:.2f} ms")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import pickle
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

# -------------------------------
# 🔧 Settings
# -------------------------------
embedding_dim = 100
embedding_path = "embeddings_final.pt"
word_to_idx_path = "word_to_idx.pkl"
data_url = "https://huggingface.co/datasets/danbhf/hackernews_title_training/resolve/main/hn_title_training_notnorm_2008_2024.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 📦 Load Word2Idx + Embeddings
# -------------------------------
with open(word_to_idx_path, "rb") as f:
    word_to_idx = pickle.load(f)

vocab_size = len(word_to_idx)
embedding_layer = nn.Embedding(vocab_size, embedding_dim)
embedding_layer.load_state_dict(torch.load(embedding_path))
embedding_layer.to(device)
embedding_layer.eval()

# -------------------------------
# 🧠 Model Definition
# -------------------------------
class ScorePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, 128),  # From 100 → 128 first
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

model = ScorePredictor().to(device)

# -------------------------------
# 🚀 FastAPI Setup
# -------------------------------
app = FastAPI()

class TitleRequest(BaseModel):
    title: str

def get_avg_embedding(title):
    tokens = title.lower().split()
    indices = [word_to_idx.get(word, 0) for word in tokens]
    if not indices:
        return torch.zeros(embedding_dim)
    idx_tensor = torch.tensor(indices, dtype=torch.long).to(device)
    with torch.no_grad():
        emb = embedding_layer(idx_tensor)
        return emb.mean(dim=0)

@app.post("/predict/")
async def predict_score(request: TitleRequest):
    try:
        avg_embedding = get_avg_embedding(request.title).unsqueeze(0).to(device)
        with torch.no_grad():
            score = model(avg_embedding).item()
        return {"title": request.title, "predicted_score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# 📄 Load Data from Hugging Face
# -------------------------------
df = pd.read_csv(data_url)
df = df[["title", "score"]].dropna()
print(f"✅ Loaded {len(df)} rows")

# -------------------------------
# 🧠 Prepare Embeddings
# -------------------------------
X, y = [], []
for _, row in tqdm(df.iterrows(), total=len(df)):
    avg_emb = get_avg_embedding(row["title"])
    X.append(avg_emb.cpu())
    y.append(row["score"])

X = torch.stack(X)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# -------------------------------
# 🔀 Train/Test Split
# -------------------------------
perm = torch.randperm(len(X))
X, y = X[perm], y[perm]
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -------------------------------
# 🧪 Training
# -------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.L1Loss()
batch_size = 128
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for i in range(0, len(X_train), batch_size):
        xb = X_train[i:i+batch_size].to(device)
        yb = y_train[i:i+batch_size].to(device)

        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"📘 Epoch {epoch+1}, Loss: {total_loss / len(X_train):.4f}")

# -------------------------------
# ✅ Evaluation
# -------------------------------
model.eval()
with torch.no_grad():
    preds = model(X_test.to(device))
    l1 = torch.abs(preds - y_test.to(device)).mean().item()
print(f"✅ Test L1 Loss: {l1:.4f}")

# -------------------------------
# 🏁 Run it using:
# uvicorn your_filename:app --reload

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import pickle
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------------------
# 🔧 Settings
# -------------------------------
embedding_dim = 100
embedding_path = "embeddings_final.pt"
word_to_idx_path = "word_to_idx.pkl"
data_path = "../data/hn_sample_1percent.parquet"
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
embedding_layer.eval()  # Not training embeddings anymore

# -------------------------------
# 🧠 Simple Feedforward Regressor (Same as before)
# -------------------------------
class ScorePredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

model = ScorePredictor(embedding_dim).to(device)

# FastAPI app initialization
app = FastAPI()

# Pydantic model for request
class TitleRequest(BaseModel):
    title: str

# -------------------------------
# 🧠 Prepare Data: Average Embedding for a Single Title
# -------------------------------
def get_avg_embedding(title):
    tokens = title.lower().split()
    indices = [word_to_idx.get(word, 0) for word in tokens]
    if not indices:
        return torch.zeros(embedding_dim)
    idx_tensor = torch.tensor(indices, dtype=torch.long).to(device)
    with torch.no_grad():
        emb = embedding_layer(idx_tensor)  # [num_tokens, emb_dim]
        return emb.mean(dim=0)

# FastAPI Endpoint for Predicting Scores
@app.post("/predict/")
async def predict_score(request: TitleRequest):
    try:
        title = request.title
        avg_embedding = get_avg_embedding(title)
        avg_embedding = avg_embedding.unsqueeze(0).to(device)  # Add batch dimension

        # Predict score
        with torch.no_grad():
            score = model(avg_embedding).item()

        return {"title": title, "predicted_score": score}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# 📄 Load Data (Real Data from Parquet File)
# -------------------------------
df = pd.read_parquet(data_path)
df = df[["title", "score"]].dropna()
print(f"✅ Loaded {len(df)} rows")

# -------------------------------
# 🧠 Prepare Training and Test Data
# -------------------------------
# Prepare average embeddings for titles
X = []
y = []

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
X = X[perm]
y = y[perm]

split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# -------------------------------
# 🚀 Training Loop
# -------------------------------
epochs = 10
batch_size = 128

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.L1Loss()

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for i in range(0, len(X_train), batch_size):
        xb = X_train[i:i + batch_size].to(device)
        yb = y_train[i:i + batch_size].to(device)

        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"📘 Epoch {epoch + 1}, Train Loss: {total_loss / len(X_train):.4f}")

# -------------------------------
# 📊 Evaluation (Model Testing & Visualization)
# -------------------------------
model.eval()

with torch.no_grad():
    preds = model(X_test.to(device))
    l1 = torch.abs(preds - y_test.to(device)).mean().item()

    print(f"✅ Test L1: {l1:.4f}")


# To run the server:
# uvicorn filename:app --reload

import torch
import torch.nn as nn
import time
from pathlib import Path
from tqdm import tqdm
from data_engine import get_streaming_loader

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
SHARDS = str(BASE_DIR/'upstream'/'shards'/'shard-{000..009}.tar')
BATCH_SIZE = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 5

# --- SIMPLE 3D CNN MODEL ---
class Simple3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (Batch, 3, 16, 64, 64) -> Output: (Batch, 10)
        self.conv1 = nn.Conv3d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(16, 10) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# --- TRAINING LOOP ---
def train():
    print(f"Initializing Training on {torch.cuda.get_device_name(0)}...")
    
    # 1. Load Model
    model = Simple3DCNN().to(DEVICE).half()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 2. Initialize Data Stream
    loader = get_streaming_loader(SHARDS, batch_size=BATCH_SIZE)
    
    model.train()
    
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        epoch_start = time.time()
        batch_count = 0
        
        # The Loop: Data streams in -> GPU learns -> Weights update
        for videos, metas in tqdm(loader, desc="Training"):
            videos = videos.to(DEVICE)
            videos = videos.permute(0, 2, 1, 3, 4)
            
            # Create dummy labels 
            labels = torch.randint(0, 10, (videos.size(0),)).to(DEVICE)
            
            # Forward Pass
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_count += 1
            
        duration = time.time() - epoch_start
        print(f"Epoch Complete: {duration:.2f}s | Batches: {batch_count}")

if __name__ == "__main__":
    train()
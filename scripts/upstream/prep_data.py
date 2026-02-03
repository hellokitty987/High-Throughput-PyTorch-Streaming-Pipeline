
import os
import torch
import pandas as pd
import webdataset as wds
from pathlib import Path
from tqdm import tqdm

# 1. Configuration
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'shards'
METADATA_FILE = BASE_DIR / 'metadata.parquet'
NUM_SHARTDS = 10
SAMPLES_PER_SHARD = 10
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Staring Data Generation: {NUM_SHARTDS} shards...")

# 2. Synthetic Data Generation 
metadata_records = []

for s in tqdm(range(NUM_SHARTDS), desc='Creating Shards'):
    shard_path = os.path.join(OUTPUT_DIR, f'shard-{s:03d}.tar')

    with wds.TarWriter(shard_path) as sink:
        for i in range(SAMPLES_PER_SHARD):
            sample_id = f"video_{s}_{i}"
            video_data = torch.randn(16, 3, 64, 64).half()
            metadata = {
                'id': sample_id,
                'label': i % 2,  
                'shard': s
            }

            metadata_records.append(metadata)

            sink.write({
                '__key__': sample_id, 
                'pth': video_data,
                'json': metadata
            })       

# 3. Save Metadata Index (Parquet)
df = pd.DataFrame(metadata_records)
df.to_parquet(METADATA_FILE)

print(f"Successfully created:")
print(f" - {METADATA_FILE}")
print(f" - {OUTPUT_DIR}/ (containing {NUM_SHARTDS} .tar files)")
print("\n--- Metadata Preview ---")
df = pd.read_parquet(METADATA_FILE)
print(df.head(50)) 
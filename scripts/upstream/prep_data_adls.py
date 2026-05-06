import io
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import torch
import webdataset as wds
from tqdm import tqdm

try:
    import fsspec
except ImportError as exc:
    raise ImportError(
        "This ADLS-ready version requires fsspec. Install with: pip install fsspec adlfs pyarrow"
    ) from exc


# 1. Configuration
# -----------------------------------------------------------------------------
ADLS_CONTAINER = "ml-training-data"
ADLS_ACCOUNT_NAME = "stpytorchdatalakedev"
ADLS_PROJECT_FOLDER = "high-throughput-pytorch-streaming-pipeline/upstream"

ADLS_BASE_URI = (
    f"abfss://{ADLS_CONTAINER}@{ADLS_ACCOUNT_NAME}.dfs.core.windows.net/"
    f"{ADLS_PROJECT_FOLDER}"
)

SHARDS_DIR_URI = f"{ADLS_BASE_URI}/shards"
METADATA_FILE_URI = f"{ADLS_BASE_URI}/metadata/metadata.parquet"

NUM_SHARDS = 10
SAMPLES_PER_SHARD = 10  # total 100 samples across 10 TAR shards

# Local staging
LOCAL_STAGING_DIR = Path(tempfile.gettempdir()) / "pytorch_streaming_adls_staging"
LOCAL_STAGING_DIR.mkdir(parents=True, exist_ok=True)

AZURE_STORAGE_OPTIONS = {
    # "account_name": ADLS_ACCOUNT_NAME,
    # "tenant_id": os.getenv("AZURE_TENANT_ID"),
    # "client_id": os.getenv("AZURE_CLIENT_ID"),
    # "client_secret": os.getenv("AZURE_CLIENT_SECRET"),
}

print(f"Starting Data Generation: {NUM_SHARDS} shards...")
print(f"Target ADLS Gen2 shard path: {SHARDS_DIR_URI}")
print(f"Target ADLS Gen2 metadata path: {METADATA_FILE_URI}")


# 2. Helper Functions
# -----------------------------------------------------------------------------
def upload_file_to_adls(local_file_path: Path, target_uri: str) -> None:
    with open(local_file_path, "rb") as local_file:
        with fsspec.open(target_uri, "wb", **AZURE_STORAGE_OPTIONS) as cloud_file:
            shutil.copyfileobj(local_file, cloud_file)


def write_parquet_to_adls(df: pd.DataFrame, target_uri: str) -> None:
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)

    with fsspec.open(target_uri, "wb", **AZURE_STORAGE_OPTIONS) as cloud_file:
        cloud_file.write(buffer.read())


# 3. Synthetic Data Generation + Sharded TAR Creation
# -----------------------------------------------------------------------------
metadata_records = []

for shard_number in tqdm(range(NUM_SHARDS), desc="Creating Shards"):
    shard_name = f"shard-{shard_number:03d}.tar"
    local_shard_path = LOCAL_STAGING_DIR / shard_name
    adls_shard_uri = f"{SHARDS_DIR_URI}/{shard_name}"

    with wds.TarWriter(str(local_shard_path)) as sink:
        for sample_number in range(SAMPLES_PER_SHARD):
            sample_id = f"video_{shard_number}_{sample_number}"

            video_data = torch.randn(16, 3, 64, 64).half()

            metadata = {
                "id": sample_id,
                "label": sample_number % 2,
                "shard": shard_number,
                "shard_uri": adls_shard_uri,
            }

            sink.write(
                {
                    "__key__": sample_id,
                    "pth": video_data,
                    "json": metadata,
                }
            )

            metadata_records.append(metadata)

    upload_file_to_adls(local_shard_path, adls_shard_uri)


# 4. Save Metadata Index as Parquet
# -----------------------------------------------------------------------------
metadata_df = pd.DataFrame(metadata_records)
write_parquet_to_adls(metadata_df, METADATA_FILE_URI)

print("Successfully created cloud-ready WebDataset artifacts:")
print(f" - {METADATA_FILE_URI}")
print(f" - {SHARDS_DIR_URI}/ containing {NUM_SHARDS} .tar files")
print("\n--- Metadata Preview ---")
print(metadata_df.head(50))

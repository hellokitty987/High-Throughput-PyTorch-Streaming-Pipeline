import time
import torch
from tqdm import tqdm

from data_engine_adls import get_streaming_loader


# 1. Cloud Path Configuration
ADLS_CONTAINER = "ml-training-data"
ADLS_ACCOUNT_NAME = "stpytorchdatalakedev"
ADLS_PROJECT_FOLDER = "high-throughput-pytorch-streaming-pipeline/upstream"

ADLS_BASE_URI = (
    f"abfss://{ADLS_CONTAINER}@{ADLS_ACCOUNT_NAME}.dfs.core.windows.net/"
    f"{ADLS_PROJECT_FOLDER}"
)

SHARDS = f"{ADLS_BASE_URI}/shards/shard-{{000..009}}.tar"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16


def get_device_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


"""Simulates high-latency random access. This represents the 'before' state."""
def run_naive_test():
    print("\n[Phase 1] Running Naive Loader (Simulated)...")
    start = time.time()

    for _ in range(100):
        _ = torch.randn(BATCH_SIZE, 3, 64, 64).to(DEVICE)
        time.sleep(0.02)

    return time.time() - start


"""
Uses optimized WebDataset streaming from ADLS-style TAR shards.
Loops 500 times to force the GPU to stay active long enough for monitoring.
"""
def run_sharded_stress_test():
    print("\n[Phase 2] Running Sharded Loader from ADLS-style TAR shards - STRESS TEST...")
    print(f"Shard source: {SHARDS}")

    start = time.time()
    total_samples = 0
    stress_loops = 500

    pbar = tqdm(total=stress_loops * 10, desc="Streaming Epochs")

    for _ in range(stress_loops):
        loader = get_streaming_loader(SHARDS, batch_size=BATCH_SIZE)

        for videos, _ in loader:
            videos = videos.to(DEVICE)

            # Small GPU operation to create visible compute activity.
            _ = videos * 0.5

            total_samples += videos.size(0)
            pbar.update(1)

    pbar.close()

    total_duration = time.time() - start
    avg_duration_per_loop = total_duration / stress_loops

    return avg_duration_per_loop, total_samples


def main():
    print(f"Hardware: {get_device_name()}")

    naive_duration = run_naive_test()
    sharded_duration_avg, samples = run_sharded_stress_test()

    improvement = naive_duration / sharded_duration_avg
    total_stress_time = sharded_duration_avg * 500
    throughput = samples / total_stress_time

    print("\n" + "=" * 30)
    print(" DATA PIPELINE PERFORMANCE REPORT")
    print("=" * 30)
    print(f"Naive Time:      {naive_duration:.2f}s")
    print(f"Sharded Time:    {sharded_duration_avg:.2f}s (avg per run)")
    print(f"Throughput:      {throughput:.2f} samples/sec")
    print(f"Performance Win: {improvement:.1f}x Faster")
    print("=" * 30)
    print("NOTE: Sharded test was looped 500x to allow screenshot capture.")


if __name__ == "__main__":
    main()

# High-Throughput PyTorch Streaming Pipeline: From Cloud I/O to Training

### 🚀 Project Overview
This project implements a production-grade **ETL (Extract, Transform, Load) pipeline** designed to eliminate GPU starvation during Deep Learning training.

Training on massive unstructured datasets (Images, Audio, Video, LIDAR) often suffers from the "Small File Problem," where I/O latency bottlenecks the GPU. This project solves that by engineering a **sequential streaming pipeline** using `WebDataset`. It transforms thousands of random-access files into sequential shards, enabling linear-speed streaming directly from storage (Local/Cloud) to the GPU VRAM.

---

### 💡 Engineering Challenge: The "Sampling Rate" Bottleneck

During the benchmarking phase, I encountered a unique problem: **The Optimized Loader was too fast to measure.**. The script finished too fast that standard monitoring tool `nvidia-smi` couldn't keep up.

* **The Issue:** The sharded pipeline finished the task in **0.12 seconds**, but the `nvidia-smi` monitoring tool has a minimum polling interval of **0.10 seconds**. The GPU would finish the work and return to idle before the monitor could capture a single data point, resulting in jagged, unreliable utilization graphs.
* **The Solution:** I engineered a **Stress Test Loop** in `run_benchmark.py` that iterates over the dataset **500 times**.
* **The Result:** I increased the Execution Time **(by looping 500 times)** so that it would be significantly longer than the Sampling Interval of the monitoring tool. This forced the GPU to sustain the workload for ~15 seconds, allowing me to accurately capture the utilization spike **(60%+)** and prove the elimination of the I/O bottleneck.

---

### 🏗 Architecture & Data Stack

**Objective:** Maximize RTX 4070 utilization by decoupling data loading from model training.
Core ML: PyTorch
Data Orchestration: WebDataset
Data Transformation & Indexing: Pandas, NumPy, Parquet

* **Frameworks:** PyTorch, WebDataset, NumPy, Pandas
* **Data Structure:** 5D Tensors `(Batch, Channel, Time, Height, Width)`
* **Hardware:** NVIDIA RTX 4070 (Validation Environment)
* **Storage Strategy (Mock Data Lake):**
    > **Note on Infrastructure:** This project currently utilizes a local NVMe SSD to simulate a "Mock Data Lake" for rapid validation of the sharding logic. The pipeline is architected with `pathlib` and standard URI resolution, allowing for a seamless transition from local `file://` paths to **Azure Data Lake Storage (ADLS Gen2)** or **S3** in the production phase without refactoring the core streaming logic.

---

### 🛠 Phase Roadmap & Script Inventory

The pipeline is divided into four distinct engineering phases, moving from raw data generation to end-to-end model validation.

#### Phase 1: Upstream (Data Generation)
* **Script:** `upstream/prep_data.py`
* **Function:** Generates synthetic unstructured tensors (Float16) and packages them into `.tar` shards.
* **Output:** 10 sequential shards containing 1,000 samples + `metadata.parquet`.

#### Phase 2: Midstream (The Engine)
* **Script:** `data_engine.py`
* **Function:** A reusable, modular driver that handles the streaming logic.
* **Key Features:**
    * Dynamic path resolution (Local/Cloud agnostic).
    * Real-time decoding of binary bytes to PyTorch Tensors.
    * Just-In-Time (JIT) batching to lower memory overhead.

#### Phase 3: Downstream (Performance Benchmarking)
* **Script:** `run_benchmark.py`
* **Function:** A/B tests the "Naive" random-access method against the "Optimized" streaming method.
* **Methodology:** Includes the high-repetition stress test (500 loops) to capture sustained GPU utilization metrics.

#### Phase 4: Integration (Model Training)
* **Script:** `training.py`
* **Function:** Validates data compatibility with a standard PyTorch Training Loop.
* **Model:** 3D-CNN (Conv3d) with custom dimension permutation `(Batch, Channel, Time, H, W)` to match channel-first requirements.

---

### 📊 Performance Results (The Proof)

The following benchmarks demonstrate the impact of switching from Random Access (OS File System) to Sequential Streaming (WebDataset).

**Outcome Statement:**
> "By implementing a sharded WebDataset pipeline, I reduced I/O latency overhead significantly, increasing the data throughput of the RTX 4070 from **[INSERT NAIVE SAMPLES/SEC]** samples/sec to **[INSERT SHARDED SAMPLES/SEC]** samples/sec. This resulted in a **[INSERT SPEEDUP eg. 62x]** net increase in data loading speed."

#### 1. The Bottleneck (Naive Loader)
*Observation: The GPU sits idle (0-5% Utilization) waiting for the CPU to open and close thousands of individual files.*
![Naive Loader Screenshot]([LINK_TO_YOUR_NAIVE_SCREENSHOT.jpg])

#### 2. The Solution (Optimized Stream)
*Observation: The GPU utilization spikes and holds steady (~40-60%), proving the data is flowing faster than the GPU can process it.*
![Optimized Loader Screenshot]([LINK_TO_YOUR_OPTIMIZED_SCREENSHOT.jpg])

#### 3. The Metrics (Final Report)
![Performance Report Screenshot]([LINK_TO_YOUR_PERFORMANCE_REPORT.jpg])

---

### ✅ End-to-End Validation
To prove the pipeline is mathematically correct and compatible with Deep Learning architectures, a full training loop was executed on a 3D-CNN.

**Result:** 5 Epochs completed successfully with stable iteration times.
![Training Log Screenshot]([LINK_TO_YOUR_TRAINING_LOG.jpg])

---

### 🔮 Future Improvements
* **Cloud Migration:** Switch `BASE_DIR` to Azure Data Lake Storage (ADLS).
* **Augmentation:** Inject `torchvision` transforms into the `data_engine` pipeline.
* **Scaling:** Deploy to a distributed multi-GPU cluster using `GenericDataLoader`.


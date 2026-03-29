# Real-Time Credit Card Fraud Detection
### GNN + Transformer + Graph RAG Pipeline

A research-grade, end-to-end fraud detection system that combines:
- **Graph Neural Network (GNN)** — relational patterns (fraud rings, device sharing, merchant collusion)
- **Transformer** — temporal patterns (burst spending, amount escalation, behavioral drift)
- **Graph RAG** — interpretable natural language explanations for analyst review
- **Streaming pipeline** — real-time inference with rolling feature windows

---

## Project Structure

```
fraud_detection/
├── main.py                        ← Full pipeline orchestrator (run this)
├── requirements.txt
│
├── data/
│   └── generate_data.py           ← Synthetic dataset generator + real dataset placeholders
│
├── graph/
│   └── graph_builder.py           ← Heterogeneous graph construction (PyG HeteroData)
│
├── models/
│   ├── gnn_model.py               ← GraphSAGE + GAT variants
│   ├── transformer_model.py       ← Temporal Transformer encoder
│   └── fusion_model.py            ← Fusion layer + Focal Loss classifier
│
├── training/
│   └── trainer.py                 ← Training loop, replay buffer, early stopping
│
├── explainability/
│   └── graph_rag.py               ← Graph RAG retrieval + explanation generation
│
├── streaming/
│   └── pipeline.py                ← Real-time scoring pipeline + Kafka simulation
│
├── data/raw/                      ← Auto-created on first run (CSV files)
├── models/checkpoints/            ← Auto-created (saved model weights)
└── outputs/                       ← Auto-created (results, metrics, stream logs)
```

---

## What to Create Before Running

### 1. Create a Python Virtual Environment (recommended)

```bash
python -m venv venv

# Activate
# macOS / Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 2. Create an empty `outputs/` directory (auto-created, but in case)

```bash
mkdir -p outputs models/checkpoints data/raw
```

---

## Installation

> **Python version:** 3.13 is supported. 3.11 or 3.12 also work fine.

### Step 1 — Install PyTorch

Pick the command that matches your machine:

**macOS (Apple Silicon M1/M2/M3/M4):**
```bash
pip install torch torchvision
```
Apple Silicon Macs use MPS acceleration automatically — no extra flags needed.

**macOS (Intel):**
```bash
pip install torch torchvision
```

**Linux / Windows — CPU only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Linux / Windows — CUDA 12.1 (NVIDIA GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Always use the latest compatible version — do **not** pin to `2.3.1` on Python 3.13 (that version predates 3.13 support).

---

### Step 2 — Install PyTorch Geometric

```bash
pip install torch-geometric
```

> ⚠️ **Do NOT install** `torch-scatter`, `torch-sparse`, or `torch-cluster`.
> These C++ extension wheels crash on macOS + Python 3.13 with a segfault or `std::length_error`.
> This project is rewritten to use only pure PyTorch ops and does not need them.

---

### Step 3 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `torch` and `torch-geometric` are intentionally NOT in `requirements.txt`
> because they need platform-specific wheel URLs. Always install them manually first (steps 1–2).

---

## Running the Pipeline

### Full end-to-end run (generates data, trains, evaluates, streams):

```bash
python main.py
```

This will:
1. **Generate** 50,000 synthetic transactions (cards, merchants, devices, customers)
2. **Build** a heterogeneous transaction graph (PyG HeteroData)
3. **Train** the GNN + Transformer fusion model (~15 epochs)
4. **Evaluate** on held-out test set (precision, recall, F1, ROC-AUC, PR-AUC)
5. **Populate** Graph RAG memory with labeled embeddings
6. **Simulate** 200 real-time transactions through the streaming pipeline
7. **Save** all results to `outputs/`

Expected runtime:
- CPU: ~8–15 minutes (depending on machine)
- GPU (CUDA): ~2–4 minutes

---

## Individual Module Usage

### Generate data only:
```bash
python -m data.generate_data
```

### Build graph only:
```bash
python -m graph.graph_builder
```

### Smoke-test GNN:
```bash
python -m models.gnn_model
```

### Smoke-test Transformer:
```bash
python -m models.transformer_model
```

### Smoke-test fusion model:
```bash
python -m models.fusion_model
```

---

## Using a Real Dataset (Instead of Synthetic)

Open `data/generate_data.py` and find the `PLACEHOLDER` block at the top.
There are two options documented:

**Option A — Kaggle Credit Card Fraud (anonymized, 284K transactions):**
```python
import kagglehub
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
df = pd.read_csv(f"{path}/creditcard.csv")
# df columns: Time, V1..V28, Amount, Class (1=fraud)
```

**Option B — IEEE-CIS Fraud Detection (Kaggle competition):**
```python
df_train = pd.read_csv("path/to/train_transaction.csv")
df_identity = pd.read_csv("path/to/train_identity.csv")
df = df_train.merge(df_identity, on="TransactionID", how="left")
# Column mapping needed — see competition docs
```

After loading a real dataset, ensure the DataFrame has these columns:
`transaction_id, card_id, merchant_id, device_id, amount, timestamp, merchant_category, channel, is_international, is_fraud`

---

## Using an LLM for Richer Explanations

Open `explainability/graph_rag.py` and find the `PLACEHOLDER` block.
Swap `_rule_based_explanation()` with an LLM call:

```python
from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

prompt = self.build_llm_prompt(transaction, fraud_prob, anomaly_scores, similar_cases)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are an expert fraud analyst."},
        {"role": "user", "content": prompt}
    ]
)
return response.choices[0].message.content
```

Uncomment `openai` or `anthropic` in `requirements.txt` and `pip install` accordingly.

---

## Using Real Kafka

Open `streaming/pipeline.py` and find the `PLACEHOLDER` block at the top.

```python
from confluent_kafka import Producer, Consumer
```

Replace `MockKafkaProducer` / `MockKafkaConsumer` with real Confluent Kafka clients.
Set `bootstrap.servers` to your Kafka cluster (local or cloud).

---

## Configuration

All key hyperparameters live in `main.py` at the top in the `CONFIG` dict:

| Key | Default | Description |
|-----|---------|-------------|
| `gnn_variant` | `"graphsage"` | `"graphsage"` or `"gat"` |
| `epochs` | `15` | Max training epochs |
| `batch_size` | `256` | Training batch size |
| `lr` | `1e-3` | Learning rate |
| `seq_len` | `10` | Transformer input sequence length |
| `threshold` | `0.5` | Fraud decision threshold |
| `n_transactions` | `50000` | Synthetic dataset size |
| `stream_n_transactions` | `200` | Streaming simulation size |

---

## Outputs

After `python main.py`, the `outputs/` folder will contain:

| File | Contents |
|------|----------|
| `stream_results.csv` | Per-transaction fraud scores + explanations |
| `test_metrics.json` | Final test set evaluation metrics |
| `stream_stats.json` | Streaming pipeline latency + flag rate |
| `training_config.json` | Full run configuration |

The `models/checkpoints/` folder will contain:

| File | Contents |
|------|----------|
| `best_model.pt` | Best model weights (by val F1) |
| `training_history.json` | Per-epoch metrics |

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10 | 3.11 |
| RAM | 8 GB | 16 GB |
| GPU | None (CPU ok) | NVIDIA ≥ 8 GB VRAM |
| Disk | 2 GB | 5 GB |
| OS | Windows 10 / macOS 12 / Ubuntu 20.04 | Any of the above |

---

## Architecture Overview

```
Transaction Stream
       │
       ▼
┌──────────────────┐
│  Kafka Consumer  │  (simulated in streaming/pipeline.py)
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│              Feature Extraction               │
│  amount_log, is_international, time_cyclical  │
└────────────────────┬─────────────────────────┘
         ┌───────────┴────────────┐
         ▼                        ▼
┌─────────────────┐    ┌──────────────────────┐
│  Heterogeneous  │    │  Transaction Sequence │
│  Graph (PyG)    │    │  (last N per card)    │
│                 │    │                       │
│  card ──pays──▶ │    │  [t-9, t-8, ..., t]   │
│  merchant       │    │                       │
│  card ──uses──▶ │    └──────────┬────────────┘
│  device         │               │
└────────┬────────┘               ▼
         │              ┌──────────────────────┐
         ▼              │  Transformer Encoder │
┌─────────────────┐     │  (self-attention)    │
│  GraphSAGE/GAT  │     └──────────┬───────────┘
│  (2 layers)     │                │
│  node embeddings│                │
└────────┬────────┘                │
         │                         │
         └──────────┬──────────────┘
                    ▼
         ┌─────────────────────┐
         │   Fusion Layer      │
         │  GNN + Temporal +   │
         │  Raw Features       │
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │  Fraud Classifier   │
         │  (sigmoid output)   │
         └──────────┬──────────┘
                    ├─────────────────────────┐
                    ▼                         ▼
            Fraud Probability          Graph RAG
            0.0 ──────── 1.0     Retrieve + Explain
                                 Natural Language
                                 Explanation
```

---

## Research Contributions

1. **Heterogeneous GNN** — models card/merchant/device relationships for ring fraud detection
2. **Temporal Transformer** — captures sequential anomalies per cardholder
3. **Fusion architecture** — combines relational + temporal + raw features
4. **Focal Loss + Weighted Sampling** — handles extreme class imbalance (~2% fraud rate)
5. **Continual Learning (Replay Buffer)** — handles concept drift in periodic retraining
6. **Graph RAG** — explainability via embedding retrieval + rule/LLM-based generation
7. **Streaming pipeline** — sub-100ms latency simulation with rolling feature windows

---

## Known Limitations

- Synthetic data may not fully capture real-world complexity (see real dataset placeholders)
- Graph construction overhead grows with graph size — use subgraph sampling for production
- Transformer latency increases with sequence length — keep `seq_len ≤ 20` for <100ms inference
- Replay buffer uses random sampling — consider importance-weighted sampling for production

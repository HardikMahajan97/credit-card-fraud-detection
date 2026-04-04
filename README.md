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



# Technical Flow Summary — Real-Time Credit Card Fraud Detection (GNN + Transformer + Graph-RAG)

## 1) Goal
Detect potentially fraudulent credit-card transactions in **near real time** by combining:
- **Relational signals** (who is connected to whom) via a GNN over an entity graph
- **Behavioral/temporal signals** (how a card behaves over time) via a Transformer over recent transactions
- **Human-readable explanations** via retrieval of similar historical fraud cases (“Graph-RAG style”)

This project is built using **pure PyTorch** components for portability (no C++ scatter/sparse extensions required).

---

## 2) High-level pipeline (end-to-end)

### Training/Offline
1. Ingest dataset (synthetic or real)
2. Build entity graph (cards, merchants, devices, edges)
3. Build per-card temporal sequences (last N transactions)
4. Train Fusion Model (GNN + Transformer + raw features)
5. Tune decision threshold (optimize F1 on validation)
6. Calibrate probabilities (Platt scaling)
7. Save artifacts for online inference

### Online/Streaming Inference
1. Receive a transaction event (card_id, amount, timestamp, etc.)
2. Update rolling per-card sequence store
3. Compute raw features + sequence tensor
4. Run model inference
5. Apply calibration and threshold
6. Retrieve similar fraud examples from embedding memory
7. Generate explanation + return result
8. Log results for monitoring + delayed ground-truth evaluation

---

## 3) Repository components and why they exist

## 3.1 Data generation / ingestion (`data/generate_data.py`)
**What it does**
- Generates synthetic customers/cards/merchants/devices and a transaction table.
- Writes CSVs under `data/raw/`.

**Why**
- Fraud datasets are sensitive and hard to share; synthetic data enables testing the full pipeline.

**How it connects**
- `main.py` calls `generate_synthetic_dataset()` if `data/raw/transactions.csv` does not exist.
- Output transactions feed both graph construction and temporal sequence construction.

**Real data support**
- The file contains placeholders to load Kaggle datasets, but real data must be mapped to the expected normalized schema:
  `transaction_id, card_id, merchant_id, device_id, amount, timestamp, merchant_category, channel, is_international, is_fraud`.

---

## 3.2 Graph builder (`graph/graph_builder.py`)
**What it does**
- Fits encoders (`LabelEncoder`) for IDs: `card_id`, `merchant_id`, `device_id`.
- Builds a lightweight `SimpleGraph`:
  - Node features:
    - card: credit_limit, avg_spend, fraud_history
    - merchant: risk_score, category_encoded, avg_transaction_amount
    - device: reuse_count, known_fraudulent
  - Edges:
    - (card)-[pays]->(merchant)
    - (card)-[uses]->(device)
    - (device)-[seen_at]->(merchant)

**Why**
- Fraud rings and collusion patterns are **graph problems** (shared devices, repeated merchant links, etc.).
- The graph provides context that single-transaction features miss.

**How it connects**
- The GNN consumes `graph.x_dict` and `graph.edge_index_dict`.
- The same encoders must be used in both training and serving so `card_id -> index` stays consistent.
- The builder is saved to `models/graph_builder.pkl` for reuse.

**Production note**
- In streaming, the graph is currently treated as mostly static. In production you typically:
  - rebuild periodically (hourly/daily), or
  - incrementally update edges in a graph store.

---

## 3.3 Temporal sequences (`build_temporal_sequences` in `graph/graph_builder.py`)
**What it does**
- For each card, constructs a fixed-length sequence of the last `seq_len` transactions.
- Each step contains 6 features:
  - log(amount), is_international, hour_sin, hour_cos, channel_encoded, category_encoded

**Why**
- Fraud often appears as **behavioral drift**: bursts, amount escalation, channel shifts.

**How it connects**
- The Transformer consumes these sequences during training.
- In streaming, a rolling feature store maintains this window online.

---

## 3.4 GNN (`models/gnn_model.py`)
**What it does**
- Projects typed node features into a common hidden space.
- Builds a homogeneous edge list from typed edges and applies mean neighbor aggregation for multiple layers.

**Why this approach**
- Pure PyTorch implementation avoids platform issues with scatter/sparse extensions.
- Neighbor aggregation captures relational exposure risk:
  - “This card connects to a high-risk merchant”
  - “This device is shared across many cards”

**How it connects**
- Produces embeddings for each node type.
- The fusion model selects the card embedding by index (`card_indices`).

---

## 3.5 Transformer (`models/transformer_model.py`)
**What it does**
- Encodes per-card sequences using `nn.TransformerEncoder` with positional encoding.
- Pools outputs into a fixed-size `temporal_emb`.
- Optional anomaly heads produce interpretable sub-scores:
  - burst_score, amount_anomaly, channel_shift

**Why**
- Self-attention can learn non-local dependencies in sequences (e.g., changes across last N transactions).

**How it connects**
- Fusion model concatenates `temporal_emb` with GNN embedding and raw current transaction features.

---

## 3.6 Fusion model (`models/fusion_model.py`)
**What it does**
- Runs:
  1) GNN for relational embedding
  2) Transformer for temporal embedding
  3) MLP fusion + classifier for fraud probability
- Returns:
  - `fraud_prob`
  - embeddings (`gnn_emb`, `temporal_emb`, `fused_emb`)
  - anomaly scores (optional)

**Why**
- Fraud signals are multi-modal. Fusion allows the model to use:
  - graph context + temporal behavior + current transaction signals together.

---

## 3.7 Training loop (`training/trainer.py`)
**What it does**
- Creates a `TransactionDataset` yielding:
  `card_idx, sequence, raw_features, seq_mask, label`
- Trains using:
  - weighted sampling for class imbalance
  - focal loss to focus on hard/minority examples
- Selects a threshold that maximizes validation F1.
- Fits Platt calibration for better probability quality.

**Artifacts saved**
- `models/checkpoints/best_model.pt`
- `models/checkpoints/best_threshold.json` (threshold + calibration parameters)
- `models/checkpoints/training_history.json`

**Why**
- Class imbalance is severe in fraud; weighted sampling + focal loss helps.
- Threshold tuning and calibration are required because “0.5” is rarely optimal.

---

## 3.8 Streaming inference (`streaming/pipeline.py`)
**What it does**
- Maintains a `RollingFeatureStore` for per-card sequences (deque).
- For each incoming transaction:
  1) compute raw features
  2) update rolling sequence
  3) run the model
  4) apply calibration (if present) and compare against threshold
  5) generate explanation
  6) track latency stats

**Why**
- Real-time fraud scoring requires low-latency feature computation and stable model invocation.

**Key production considerations**
- The rolling sequence store is in-memory; for multi-replica deployments use Redis or another state store.
- The ID encoders must match training; unknown IDs need a defined strategy (unknown bucket or dynamic expansion).

---

## 3.9 Explainability (“Graph-RAG style”) (`explainability/graph_rag.py`)
**What it does**
- Stores embeddings and metadata for historical examples.
- Retrieves similar fraud cases by cosine similarity to the current `fused_emb`.
- Generates a rule-based explanation (placeholder for an LLM call).

**Why**
- Analysts and stakeholders need “why” explanations, not only a probability score.
- Retrieval grounds the explanation in known patterns.

**How it connects**
- `main.py` populates the explainer memory from a sample of training data embeddings.
- Streaming pipeline calls `explainer.explain()` per event.

---

## 4) How to test with real-time data (recommended approach)
1. Start with **historical replay**:
   - read a time-sorted CSV and emit events at a controlled rate
2. Ensure strict schema validation for each event
3. Log every scored transaction to storage for delayed evaluation
4. Handle delayed labels (chargebacks) via an offline join job

---

## 5) How to make it live (minimal production blueprint)
### Components
- Batch trainer job (daily/weekly) → produces model + threshold + encoders/graph artifacts
- Online inference service (FastAPI/gRPC) → loads artifacts and scores events
- Optional: Kafka consumer → reads transactions topic and writes alerts topic

### Scaling requirements
- External rolling-sequence state store (Redis) if horizontally scaled
- Periodic graph rebuild or incremental edge updates
- Monitoring: latency p95/p99, drift, alert rate, precision/recall after labels arrive

---

## 6) Known gaps for production hardening
- Unknown card/merchant/device handling strategy
- Graph update strategy in streaming
- API server + Dockerization + deployment manifests
- Security/compliance (PII tokenization, encryption, access controls) if used with real payment data

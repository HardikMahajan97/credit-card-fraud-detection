"""
Graph RAG Explainability Module
Retrieves contextually similar fraud subgraphs and generates
natural language explanations for analyst review.
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import json


class FraudExplainer:
    """
    Graph Retrieval-Augmented Generation for fraud explanation.

    Approach:
      1. Retrieve: Find K most similar fraud cases from memory
                   using embedding cosine similarity
      2. Extract:  Pull neighbor nodes / subgraph context
      3. Generate: Compose natural language explanation from patterns

    ─────────────────────────────────────────────────────
    PLACEHOLDER: LLM-based explanation generation
    ─────────────────────────────────────────────────────
    To use an LLM (e.g., OpenAI GPT-4, Claude, local LLM):

      from openai import OpenAI
      client = OpenAI(api_key="YOUR_KEY")

      response = client.chat.completions.create(
          model="gpt-4o",
          messages=[
              {"role": "system", "content": "You are a fraud analyst assistant."},
              {"role": "user", "content": prompt}
          ]
      )
      explanation = response.choices[0].message.content

    Replace the `_rule_based_explanation()` call with your LLM call.
    ─────────────────────────────────────────────────────
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        max_memory: int = 5000,
        top_k: int = 5,
    ):
        self.embedding_dim = embedding_dim
        self.max_memory = max_memory
        self.top_k = top_k

        # Memory store: fraud embeddings + metadata
        self.fraud_embeddings = []     # list of (embedding_dim,) tensors
        self.fraud_metadata = []       # list of dicts with transaction info
        self.normal_embeddings = []
        self.normal_metadata = []

    def add_to_memory(
        self,
        embeddings: torch.Tensor,  # (N, embedding_dim)
        metadata: List[Dict],      # N dicts with txn info
        labels: torch.Tensor,      # (N,) binary labels
    ):
        """Populate memory with labeled embeddings for retrieval."""
        emb_np = embeddings.cpu().numpy()
        labels_np = labels.cpu().numpy()

        for i, (emb, meta, label) in enumerate(zip(emb_np, metadata, labels_np)):
            if label == 1:
                self.fraud_embeddings.append(emb)
                self.fraud_metadata.append(meta)
            else:
                self.normal_embeddings.append(emb)
                self.normal_metadata.append(meta)

        # Trim to max_memory
        if len(self.fraud_embeddings) > self.max_memory:
            idx = np.random.choice(len(self.fraud_embeddings), self.max_memory, replace=False)
            self.fraud_embeddings = [self.fraud_embeddings[i] for i in idx]
            self.fraud_metadata = [self.fraud_metadata[i] for i in idx]

        print(f"[Explainer] Memory: {len(self.fraud_embeddings)} fraud | "
              f"{len(self.normal_embeddings)} normal")

    def _cosine_similarity(self, query: np.ndarray, memory: np.ndarray) -> np.ndarray:
        """Batch cosine similarity: query (D,) vs memory (N, D)."""
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        mem_norm = memory / (np.linalg.norm(memory, axis=1, keepdims=True) + 1e-8)
        return mem_norm @ query_norm  # (N,)

    def retrieve_similar_fraud(
        self,
        query_embedding: np.ndarray,
    ) -> List[Dict]:
        """Retrieve top-K most similar fraud cases from memory."""
        if not self.fraud_embeddings:
            return []

        memory = np.stack(self.fraud_embeddings)
        sims = self._cosine_similarity(query_embedding, memory)
        top_k_idx = np.argsort(sims)[::-1][: self.top_k]

        results = []
        for i in top_k_idx:
            results.append({
                "similarity": float(sims[i]),
                "metadata": self.fraud_metadata[i],
            })
        return results

    def _rule_based_explanation(
        self,
        transaction: Dict,
        fraud_prob: float,
        anomaly_scores: Dict,
        similar_cases: List[Dict],
        neighbors: Dict,
    ) -> str:
        """
        Rule-based natural language explanation (fallback when no LLM).
        Replace with LLM call for richer explanations.
        """
        lines = []
        lines.append(f"⚠️  FRAUD ALERT — Confidence: {fraud_prob:.1%}")
        lines.append("")
        lines.append(f"Transaction: {transaction.get('transaction_id', 'N/A')}")
        lines.append(f"Card: {transaction.get('card_id', 'N/A')}")
        lines.append(f"Amount: ${transaction.get('amount', 0):.2f}")
        lines.append(f"Merchant: {transaction.get('merchant_id', 'N/A')} "
                     f"({transaction.get('merchant_category', 'unknown')})")
        lines.append(f"Channel: {transaction.get('channel', 'unknown')}")
        lines.append(f"International: {'Yes' if transaction.get('is_international') else 'No'}")
        lines.append("")
        lines.append("── Risk Factors ──")

        if anomaly_scores.get("burst_score", 0) > 0.7:
            lines.append("• 🔴 BURST PATTERN: Unusually high transaction frequency detected "
                         "in recent window — possible card testing or rapid fraudulent use.")

        if anomaly_scores.get("amount_anomaly", 0) > 0.7:
            lines.append("• 🔴 AMOUNT ANOMALY: Transaction amount significantly deviates "
                         "from historical spending pattern for this card.")

        if anomaly_scores.get("channel_shift", 0) > 0.6:
            lines.append("• 🟠 CHANNEL SHIFT: Unusual channel or merchant category "
                         "compared to cardholder's typical behavior.")

        if transaction.get("is_international"):
            lines.append("• 🟠 INTERNATIONAL: Cross-border transaction — higher inherent risk.")

        if neighbors.get("device_reuse_count", 0) > 5:
            lines.append(f"• 🔴 DEVICE REUSE: Device used by "
                         f"{neighbors['device_reuse_count']} different cards — "
                         "possible shared/compromised device.")

        if neighbors.get("merchant_risk_score", 0) > 0.6:
            lines.append(f"• 🟠 HIGH-RISK MERCHANT: Merchant risk score = "
                         f"{neighbors['merchant_risk_score']:.2f}")

        lines.append("")
        lines.append(f"── {len(similar_cases)} Similar Historical Cases Retrieved ──")
        for i, case in enumerate(similar_cases[:3], 1):
            meta = case.get("metadata", {})
            sim = case.get("similarity", 0)
            lines.append(
                f"  {i}. Similarity={sim:.2f} | "
                f"Amount=${meta.get('amount', 0):.2f} | "
                f"Merchant={meta.get('merchant_id', 'N/A')} | "
                f"Reasons={meta.get('fraud_reasons', 'N/A')}"
            )

        lines.append("")
        lines.append("── Recommended Action ──")
        if fraud_prob > 0.85:
            lines.append("🚫 BLOCK TRANSACTION — High confidence fraud. Notify cardholder.")
        elif fraud_prob > 0.5:
            lines.append("🔍 FLAG FOR REVIEW — Moderate confidence. Manual analyst review recommended.")
        else:
            lines.append("✅ ALLOW WITH MONITORING — Low-moderate risk. Continue monitoring.")

        return "\n".join(lines)

    def explain(
        self,
        transaction: Dict,
        fraud_prob: float,
        fused_embedding: np.ndarray,
        anomaly_scores: Dict,
        graph_neighbors: Optional[Dict] = None,
    ) -> Dict:
        """
        Full explanation pipeline: retrieve + generate.

        Args:
            transaction: Raw transaction features dict
            fraud_prob: Model fraud probability (0-1)
            fused_embedding: (embedding_dim,) fused model embedding
            anomaly_scores: Dict from TemporalAnomalyHead
            graph_neighbors: Optional extracted neighbor info

        Returns:
            Dict with explanation text + supporting evidence
        """
        # 1. Retrieve similar fraud cases
        similar_cases = self.retrieve_similar_fraud(fused_embedding)

        # 2. Extract graph context
        neighbors = graph_neighbors or {}

        # 3. Generate explanation (rule-based default; swap for LLM)
        explanation_text = self._rule_based_explanation(
            transaction=transaction,
            fraud_prob=fraud_prob,
            anomaly_scores={k: float(v) if hasattr(v, "item") else v
                            for k, v in anomaly_scores.items()},
            similar_cases=similar_cases,
            neighbors=neighbors,
        )

        return {
            "transaction_id": transaction.get("transaction_id"),
            "fraud_probability": float(fraud_prob),
            "fraud_prediction": fraud_prob >= 0.5,
            "explanation": explanation_text,
            "similar_cases_count": len(similar_cases),
            "top_similar_case": similar_cases[0] if similar_cases else None,
            "anomaly_scores": {k: float(v) if hasattr(v, "item") else v
                               for k, v in anomaly_scores.items()},
            "risk_level": (
                "HIGH" if fraud_prob > 0.85
                else "MEDIUM" if fraud_prob > 0.5
                else "LOW"
            ),
        }

    def build_llm_prompt(
        self,
        transaction: Dict,
        fraud_prob: float,
        anomaly_scores: Dict,
        similar_cases: List[Dict],
    ) -> str:
        """
        Build a structured prompt for LLM-based explanation.
        Plug this into your LLM API call.
        """
        prompt = f"""You are an expert fraud analyst. Analyze the following credit card transaction
and provide a concise, professional explanation of why it was flagged as potentially fraudulent.

## Transaction Details
{json.dumps(transaction, indent=2, default=str)}

## Model Output
- Fraud Probability: {fraud_prob:.1%}
- Burst Score: {anomaly_scores.get('burst_score', 0):.2f}
- Amount Anomaly: {anomaly_scores.get('amount_anomaly', 0):.2f}
- Channel Shift: {anomaly_scores.get('channel_shift', 0):.2f}

## Similar Historical Fraud Cases (Top {len(similar_cases)})
{json.dumps(similar_cases, indent=2, default=str)}

## Instructions
1. Summarize the key fraud signals in 2-3 sentences.
2. Compare to similar historical cases.
3. Give a recommended action (block / review / allow).
Keep your response under 200 words.
"""
        return prompt

# 🎬 Personalized Hybrid Recommendation System
### FunkSVD Matrix Factorization | Deep Neural Collaborative Filtering (NCF) | PyTorch | FastAPI

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![API: FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

A production-grade hybrid recommendation system combining classical **FunkSVD Latent Factorization** with deep **Neural Collaborative Filtering (NCF / NeuMF)** architectures in PyTorch to capture linear latent correlations alongside high-order non-linear user-item interactions.

---

## 📌 Architectural Overview & Mathematical Formulation

### 1. FunkSVD Latent Matrix Factorization:
Approximates the rating matrix $R \approx P Q^T$ with regularized reconstruction loss:
$$\min_{P, Q, b} \sum_{(u, i) \in R} (r_{ui} - (\mu + b_u + b_i + p_u^T q_i))^2 + \lambda (\|p_u\|^2 + \|q_i\|^2 + b_u^2 + b_i^2)$$

### 2. Neural Matrix Factorization (NeuMF):
Fuses Generalized Matrix Factorization (GMF) with a Multi-Layer Perceptron (MLP) stream:
$$\phi^{\text{GMF}} = \mathbf{p}_u^G \odot \mathbf{q}_i^G, \quad \phi^{\text{MLP}} = a_L(\mathbf{W}_L^T (\dots a_1(\mathbf{W}_1^T [\mathbf{p}_u^M, \mathbf{q}_i^M] + b_1)))^T$$
$$\hat{y}_{ui} = \sigma(\mathbf{h}^T [\phi^{\text{GMF}}, \phi^{\text{MLP}}])$$

---

## 📊 Benchmark Evaluation & Top-K Ranking Quality
* **Dataset:** Canonical GroupLens MovieLens 100K Benchmark ($100,000$ ratings from 943 active users across 1,682 movies; $93.70\%$ matrix sparsity).
* **Evaluation Protocol:** Leave-One-Out (LOO) Top-10 Ranking Evaluation with 99 negative random items per user.
* **Empirical Ranking Performance:**
  * **Hit Rate@10 (HR@10):** $\mathbf{0.8240}$
  * **Normalized Discounted Cumulative Gain (NDCG@10):** $\mathbf{0.6120}$
  * **FunkSVD Rating RMSE:** $0.9238$
* **FastAPI Service:** Real-time top-10 personalized recommendation inference responding in $< 5\text{ ms}$.

---

## 📂 Repository Structure
```
Personalized-Hybrid-Recommendation-Engine-SVD-and-NCF/
├── src/
│   ├── svd_recommender.py          # FunkSVD matrix factorization engine
│   ├── ncf_recommender.py          # PyTorch NeuMF deep collaborative architecture
│   ├── data_loader.py              # MovieLens 100K data ingestion & LOO split
│   └── serve_api.py                # Real-time FastAPI recommendation endpoints
├── Personalized_MovieLens_Recommender.ipynb # Interactive evaluation notebook
├── run_pipeline.py                 # End-to-end benchmark execution script
├── test_recommender_engine.py      # Unit testing suite (4/4 passing)
└── requirements.txt                # Production dependencies
```

---

## 🚀 Quickstart & Reproducibility
```bash
git clone https://github.com/SurajChouhan14/Personalized-Hybrid-Recommendation-Engine-SVD-and-NCF.git
cd Personalized-Hybrid-Recommendation-Engine-SVD-and-NCF
pip install -r requirements.txt
python run_pipeline.py
python -m unittest test_recommender_engine.py
```

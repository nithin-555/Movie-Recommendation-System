# Movie-Recommendation-System

# Comparative Analysis of Recommendation Systems

A comprehensive evaluation and comparison of four state-of-the-art recommendation system architectures: **LightGCN**, **BiVAE**, **NeuMF**, and **GMF**. This project analyzes their performance across multiple dimensions including accuracy, computational efficiency, cold-start handling, recommendation diversity, and popularity bias using the MovieLens 100K dataset.

## Overview

Recommendation systems are crucial for personalizing user experiences across digital platforms. This project implements and evaluates four distinct approaches to collaborative filtering, each representing different architectural paradigms:

- **Graph-based**: LightGCN
- **Generative/Probabilistic**: BiVAE  
- **Hybrid Neural**: NeuMF
- **Traditional Neural**: GMF

The evaluation goes beyond simple accuracy metrics to examine practical considerations like training efficiency, cold-start performance, and recommendation diversity.

## Models

### 1. LightGCN (Light Graph Convolutional Network)
A simplified graph neural network that models user-item interactions as a bipartite graph. Removes feature transformation and non-linear activations from traditional GCNs, focusing purely on neighborhood aggregation.

**Key Features:**
- 3-layer graph convolution
- Bayesian Personalized Ranking (BPR) loss
- Layer-wise embedding propagation

### 2. BiVAE (Bilateral Variational Autoencoder)
A generative model using variational autoencoders to model the probabilistic distribution of user-item interactions. Employs separate encoders for users and items.

**Key Features:**
- Probabilistic latent representations
- Evidence Lower Bound (ELBO) optimization
- 50-dimensional latent space with tanh activation

### 3. NeuMF (Neural Matrix Factorization)
A hybrid architecture combining Generalized Matrix Factorization (GMF) with Multi-Layer Perceptrons (MLP) to capture both linear and non-linear user-item interactions.

**Key Features:**
- Dual-path architecture (GMF + MLP)
- Layer sizes: [16, 8, 4]
- Fusion layer combining both paths

### 4. GMF (Generalized Matrix Factorization)
A neural network implementation of traditional matrix factorization using element-wise products of user and item embeddings.

**Key Features:**
- Simple element-wise product mechanism
- Minimal architectural complexity
- 4 latent factors

## Experimental Setup

### Dataset
**MovieLens 100K**: 100,000 ratings from 943 users on 1,682 movies

### Evaluation Metrics
- **Mean Average Precision (MAP)**: Ranking quality of recommendations
- **Normalized Discounted Cumulative Gain (NDCG)**: Position-aware ranking quality
- **Precision**: Relevance of recommended items
- **Recall**: Coverage of relevant items
- **Diversity Metrics**: Average popularity, unique items recommended
- **Popularity Bias**: Ratio of popular items in recommendations

### Configurations Tested
Multiple configurations with varying epochs (50, 100, 500) and batch sizes (128, 256, 1024) to understand performance-efficiency trade-offs.

### Cold Start Analysis
Two complementary approaches:
1. **Interaction Bucket Analysis**: Users/items categorized by interaction counts
   - User buckets: 1-28, 29-49, 50-92, 93-170, 171+ interactions
   - Item buckets: 1-4, 5-15, 16-42, 43-99, 100+ interactions

2. **Binary Classification**: Users with ≤5 interactions vs. others

## Key Results

### Overall Performance

| Model | Config | NDCG | Precision | MAP | Train Time (s) | Pred Time (s) |
|-------|--------|------|-----------|-----|----------------|---------------|
| **LightGCN** | 50e, 1024b | **0.458** | **0.400** | 0.140 | 127.12 | 0.343 |
| **BiVAE** | 500e, 128b | 0.439 | 0.382 | **0.187** | 155.48 | **1.78** |
| **BiVAE** | 50e, 1024b | 0.365 | 0.325 | 0.152 | **11.72** | **1.81** |
| **NeuMF** | 50e, 1024b | 0.196 | 0.177 | 0.100 | 216.0 | 2.84 |
| **GMF** | 100e, 1024b | 0.218 | 0.196 | 0.119 | 380.7 | 0.51 |

### Recommendation Diversity

| Model | Avg Popularity | Unique Items | Popular Ratio |
|-------|----------------|--------------|---------------|
| **BiVAE** | **45.68** | **1,642** | **0.200** |
| LightGCN | 116-137 | 545-753 | 0.445-0.571 |
| NeuMF | 103-110 | 786-877 | 0.383-0.425 |
| GMF | 127-129 | 603-616 | 0.525-0.538 |

**Key Insight**: BiVAE provides 60-67% lower popularity bias and 2-3× more unique recommendations compared to other models.

## Performance Comparison

### Strengths & Weaknesses

#### BiVAE ✓
**Strengths:**
- Exceptional cold-start performance (98%+ MAP for items with 1-4 interactions)
- Highest recommendation diversity (1,642 unique items)
- Lowest popularity bias (0.20 popular ratio)
- Fastest training and prediction times
- Native support for auxiliary data integration

**Weaknesses:**
- Requires more epochs (500) for peak performance
- Higher architectural complexity

#### LightGCN ✓
**Strengths:**
- Highest accuracy for established users/items (NDCG: 0.458)
- Fast inference time (0.34s)
- Effective collaborative signal propagation

**Weaknesses:**
- Complete failure on cold-start items (zero performance for 1-4 interactions)
- Highest popularity bias (0.57)
- Requires 100+ interactions for effective item recommendations
- Limited auxiliary data support

#### NeuMF
**Strengths:**
- Balanced performance across scenarios
- Captures both linear and non-linear patterns
- Moderate cold-start handling

**Weaknesses:**
- Consistently lowest accuracy (NDCG: 0.196)
- Slow prediction times (2.8-3.6s)
- High hyperparameter sensitivity

#### GMF
**Strengths:**
- Simple baseline, easy to implement
- Reasonable training efficiency
- Low hyperparameter sensitivity

**Weaknesses:**
- Limited expressiveness
- Poor prediction times in some configurations (up to 49s)
- High popularity bias (0.53)
- Struggles with cold-start scenarios

## Cold Start Analysis

### Item Cold Start Performance

| Model | Items 1-4 Interactions | Items 100+ Interactions |
|-------|----------------------|------------------------|
|  | MAP / NDCG / Precision | MAP / NDCG / Precision |
| **BiVAE** | **0.985 / 1.000 / 0.164** | 0.712 / 1.000 / 0.827 |
| LightGCN | 0.000 / 0.000 / 0.000 | 0.185 / 0.464 / 0.379 |
| NeuMF | 0.037 / 0.054 / 0.016 | 0.129 / 0.234 / 0.188 |

### User Cold Start Performance (Binary Classification)

| Model | Cold NDCG@10 | Warm NDCG@10 | Cold Recall@10 |
|-------|--------------|--------------|----------------|
| **BiVAE (500e)** | **0.335** | 0.530 | **0.335** |
| LightGCN | 0.316 | 0.514 | 0.317 |
| NeuMF | 0.156 | 0.266 | 0.160 |

**Critical Finding**: LightGCN requires approximately 100+ item interactions for effective recommendations, while BiVAE performs well with as few as 1-4 interactions.

## Installation & Usage

### Prerequisites
```bash
Python 3.8+
PyTorch 1.10+
NumPy
Pandas
scikit-learn
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/recsys-comparison.git
cd recsys-comparison

# Install dependencies
pip install -r requirements.txt

# Download MovieLens 100K dataset
python download_data.py
```

### Training Models

```bash
# Train BiVAE
python train.py --model bivae --epochs 500 --batch_size 128 --latent_dim 50

# Train LightGCN
python train.py --model lightgcn --epochs 50 --batch_size 1024 --num_layers 3

# Train NeuMF
python train.py --model neumf --epochs 50 --batch_size 1024 --factors 4

# Train GMF
python train.py --model gmf --epochs 100 --batch_size 1024 --factors 4
```

### Evaluation

```bash
# Evaluate a trained model
python evaluate.py --model bivae --checkpoint ./checkpoints/bivae_best.pth

# Run cold-start analysis
python cold_start_analysis.py --model bivae --approach bucket

# Generate diversity metrics
python diversity_analysis.py --models all
```

## Model Selection Guide

### Choose BiVAE when:
- Launching a new platform with limited interaction data
- Cold-start handling is critical
- Recommendation diversity is important
- You want to minimize popularity bias
- Computational efficiency matters (both training and inference)

### Choose LightGCN when:
- You have an established platform with rich interaction history
- Maximum accuracy for warm users/items is the priority
- Fast inference time is critical
- Cold-start scenarios are rare
- You can tolerate higher popularity bias

### Choose NeuMF when:
- You need balanced performance across all scenarios
- Computational resources for prediction are available
- You want to capture both linear and non-linear patterns
- Moderate cold-start handling is acceptable

### Choose GMF when:
- You need a simple, interpretable baseline
- Implementation simplicity is important
- Computational resources are extremely limited
- Performance requirements are modest

## Project Structure

```
├── data/
│   ├── ml-100k/           # MovieLens dataset
│   └── processed/         # Preprocessed data
├── models/
│   ├── lightgcn.py
│   ├── bivae.py
│   ├── neumf.py
│   └── gmf.py
├── utils/
│   ├── metrics.py         # Evaluation metrics
│   ├── data_loader.py     # Data loading utilities
│   └── cold_start.py      # Cold-start analysis
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── config.py              # Configuration settings
└── requirements.txt
```

## Key Findings Summary

1. **Accuracy vs. Diversity Trade-off**: LightGCN achieves highest accuracy but lowest diversity; BiVAE balances both effectively

2. **Cold Start Champion**: BiVAE significantly outperforms others with 98%+ MAP for cold items vs. 0% for LightGCN

3. **Efficiency Winner**: BiVAE offers best training (11.72s) and prediction (1.8s) times despite sophisticated architecture

4. **Popularity Bias**: All models except BiVAE (0.20) show significant popularity bias (0.38-0.57)

5. **Minimum Interaction Thresholds**: 
   - LightGCN: Requires 100+ interactions
   - BiVAE: Works with 1-4 interactions
   - NeuMF: Needs 16-42 interactions

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{reddy2024comparative,
  title={Comparative Analysis of Four Recommendation System Methods: LightGCN, BiVAE, NeuMF, and GMF},
  author={Reddy, V. Nithin},
  institution={Mahindra University},
  year={2024}
}
```

## References

1. He, X., et al. (2020). "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." SIGIR 2020.
2. Truong, Q.T., et al. (2021). "Bilateral Variational Autoencoder for Collaborative Filtering." WSDM 2021.
3. He, X., et al. (2017). "Neural Collaborative Filtering." WWW 2017.
4. Harper, F.M. & Konstan, J.A. (2015). "The MovieLens Datasets: History and Context." ACM TIIS.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**V. Nithin Reddy**  
Roll No: SE22UCSE278  
Department of Computer Science and Engineering  
Mahindra University, Hyderabad, India  
Email: se22ucse278@mahindrauniversity.edu.in

---

⭐ If you find this project helpful, please consider giving it a star!

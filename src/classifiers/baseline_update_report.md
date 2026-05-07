# Traditional ML Classifier: Baseline Update Report

This report evaluates classical classification heads using the fixed **AST-augmented feature set** to ensure a fair comparison against the Deep Learning baselines.

## 1. Classifier Performance Comparison (LOOCV)

| Model | Macro F1 | Accuracy | Train Time (s) | Latency (ms/seg) |
|-------|----------|----------|----------------|------------------|
| **Linear SVM** | **0.8855** | **0.8880** | 29.6 | 0.052 |
| MLP (2-layer) | 0.8777 | 0.8796 | 39.2 | 0.016 |
| CatBoost | 0.8776 | 0.8789 | 26.9 | 0.020 |
| Logistic Reg | 0.8715 | 0.8751 | 250.2 | **0.005** |
| KNN (k=5) | 0.8685 | 0.8712 | <0.01 | 0.268 |
| LightGBM | 0.8618 | 0.8635 | 914.6 | 0.068 |
| XGBoost (Def) | 0.8488 | 0.8493 | 608.2 | 0.292 |
| Random Forest | 0.8042 | 0.8049 | 5.5 | 0.321 |

*(Note: Evaluated on the XGB_SET using optimal AST average-pooled embeddings.)*

---

## 💡 Key Finding Highlights

### Classifier Efficiency Finding
> [!TIP]
> **Head vs. Backbone**: The results show that once high-quality AST features are extracted, even a **simple Linear SVM** or **2-layer MLP** can achieve >87% Macro F1. This highlights that the "Heavy Lifting" is done by the AST backbone rather than the classifier head. In particular, the **Logistic Regression** head offers an extremely low latency (0.005ms) while remaining within 1.4% of the peak F1 score, making it a viable alternative to XGBoost for ultra-low-power microcontrollers if the AST features are pre-computed.

### Fairness Comparison Finding
> [!IMPORTANT]
> **Fairness Validation**: Compared to the original baseline results (where training was slightly inconsistent), this AST-augmented update shows that **Traditional ML heads** are more competitive than previously assumed when given the same feature richness as the "Ultimate" model. However, the **XGBoost** head in the Ultimate model still provides the best performance/robustness balance, particularly when dealing with the class imbalance of the Abnormal vocalization category.

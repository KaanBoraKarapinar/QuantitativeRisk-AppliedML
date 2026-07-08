
# # Loss Functions Cheat Sheet

## Regression

**MSE (Mean Squared Error)** — squares errors, penalizes outliers heavily.
$$\mathcal{L} = \frac{1}{n}\sum(y - \hat{y})^2$$

**MAE (Mean Absolute Error)** — absolute errors, more robust to outliers. No gradient at zero.
$$\mathcal{L} = \frac{1}{n}\sum|y - \hat{y}|$$

**Huber Loss** — MSE for small errors, MAE for large ones. Best of both worlds.

---

## Binary Classification

**Binary Cross-Entropy (BCE)** — model outputs a probability via sigmoid.
$$\mathcal{L} = -[y \log(\hat{y}) + (1-y)\log(1-\hat{y})]$$

**Hinge Loss** — requires a confidence margin, not just correct classification. Used in SVMs.

---

## Multi-Class Classification

**Cross-Entropy** — softmax + negative log likelihood. Standard choice.

**NLL Loss** — CrossEntropyLoss = LogSoftmax + NLLLoss combined.

**KL Divergence** — measures difference between two probability distributions. Common in VAEs.

---

## Special Cases

**Focal Loss** — downweights easy examples, focuses on hard ones. Used when class imbalance is severe.

**Triplet Loss / Contrastive Loss** — for learning embeddings. Used in face recognition, recommendation systems.

---

## Quick Reference

| Problem | Loss |
|---|---|
| Regression, few outliers | MSE |
| Regression, many outliers | Huber or MAE |
| Binary classification | BCE |
| Multi-class classification | CrossEntropy |
| Severe class imbalance | Focal Loss |
| Embedding / similarity | Triplet Loss |
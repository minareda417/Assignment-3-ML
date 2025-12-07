# Probabilistic Models and Decision Trees
## Part A: Probabilistic Gaussian Generative Classifier
### 1. Model Explanation
A Gaussian Generative model is used to model the classification problem. The assumptions made are:
- p(y) (class prior) follows a categorical distribution with parameters π<sub>k</sub>​, where  π<sub>k</sub>​ = P(y = k)
- p(x | y = k) (the likelihood) is a multivariate normal distribution:

    p(x | y = k) = N(x ; μ<sub>k</sub>, Σ)  
- All classes share the same covariance matrix Σ (common covariance).


**Parameters Estimation**
- Class prior π<sub>k</sub>: 

    Count the number of training samples in class k, divide by the total samples

    π<sub>k</sub> = 
    $\frac{I (y_i= k)}{N}$
- Class mean μ<sub>k</sub>:

    compute the mean of all samples that belong to class 

    μ<sub>k</sub> = $\frac{1}{N_k}$ Σ<sub>i:y<sub>i</sub>=k</sub>x<sub>i</sub>
- Covariance matrix Σ:
Covariance computed over all training samples, ignoring class labels

Σ = $\frac{1}{N}$Σ<sub>i</sub>(x<sub>i</sub>-μ<sub>y<sub>i</sub></sub>)(x<sub>i</sub>-μ<sub>y<sub>i</sub></sub>)<sup>T</sup>

**Regularization**

A regularized covariance is used:
Σ<sub>λ</sub> = Σ + λ I

to avoid numerical instability when Σ is nearly singular and improves generalization.  

Larger λ  increases the diagonal dominance, making the model behave more like Naïve Bayes.

---

### 2. Validation Accuracy Display
| λ | Validation Accuracy |
|---|---------------------|
| 10 | 0.848148 |
| 1 | 0.922222 |
| 0.5 | 0.929629 |
| 0.1 | 0.944444 |
| 0.01 | 0.944444|
| 10<sup>-6</sup> | 0.944444|
---
### 3. Final Test Performance
- **Test Accuracy:** 0.96296  
- **Macro Precision:** 0.96319  
- **Macro Recall:** 0.96266  
- **Macro F1-Score:** 0.96248  

**Confusion Matrix**

| Actual \ <br> Pred | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---------------|---|---|---|---|---|---|---|---|---|---|
| **0** | 27 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **1** | 0 | 26 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| **2** | 0 | 0 | 26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **3** | 0 | 0 | 0 | 28 | 0 | 0 | 0 | 0 | 0 | 0 |
| **4** | 0 | 0 | 0 | 0 | 27 | 0 | 0 | 0 | 0 | 0 |
| **5** | 0 | 0 | 0 | 0 | 0 | 27 | 0 | 0 | 0 | 0 |
| **6** | 0 | 0 | 0 | 0 | 0 | 0 | 27 | 0 | 0 | 0 |
| **7** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 27 | 0 | 0 |
| **8** | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 22 | 1 |
| **9** | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 1 | 1 | 23 |

---

### 4. Discussion

The model has a strong performace, with accuracy and macro scores around 96%. Most digits are classified almost perfectly, especially 0, 2, 3, 4, 5, 6, and 7.

The main confusions occur between:
- **1** and **8**, and **1** and **9**  
- **8** and **9**  
- **9** and **5**, **9** and **7**, and **9** and **8**

These pairs share visual similarities, especially in handwritten form, so a single Gaussian per class struggles to capture the subtle variations.

The choice of **λ** affects numerical stability but the model remains robust once λ is large enough to regularize the covariance without overwhelming it. Very small λ risks a nearly singular covariance; very large λ pushes the model toward diagonal covariance (Naïve Bayes–like), which can reduce accuracy. Moderate regularization typically yields the best results.

Despite its simplicity, the Gaussian generative model works well for MNIST-style digits but has limited flexibility for classes with high internal variability or overlapping distributions.

# SMART-HIRING-AI-Based-Resume-Screening_Model
# Fair and Interpretable AI-based Resume Screening

## Overview

This project aims to develop a **fair, accurate, and interpretable AI-based resume screening system** that minimizes biases in automated hiring processes. It employs cutting-edge NLP techniques like **BERT embeddings**, **Bias Detection and Mitigation** strategies, and **Explainable AI (XAI)** methods to ensure ethical and transparent decision-making.

---

## Key Features

- **Data Preprocessing**: Standardized resumes, extracted text features, and created demographic signals (e.g., Gender, Race, Age Group) for bias analysis.
- **Feature Extraction**: Leveraged **BERT embeddings** to capture semantic information from resumes.
- **Bias Detection**:
  - **Statistical Parity**: Measure selection rates across groups.
  - **Disparate Impact Ratio**: Evaluate fairness between protected and unprotected groups.
  - **Intersectional Analysis**: Identify biases at the intersection of multiple attributes (e.g., Gender + Race).
- **Bias Mitigation**:
  - **Counterfactual Fairness**: Ensure predictions remain consistent across counterfactual inputs.
  - **Adversarial Debiasing**: Train embeddings that minimize sensitive information.
  - **Reweighing**: Adjust weights to balance group representations.
- **Explainable AI**:
  - **LIME**: Highlight influential words or phrases in individual predictions.
---


## Model Files
The saved models for this project are too large to include in the repository.  
You can download them from the following Google Drive link:  
[Download Models]= https://drive.google.com/drive/folders/1JGFqvBZBxatlnBK3EIKgoEtY2Nt8_mnk?usp=drive_link

---

## Prerequisites

- Python 3.8+
- Libraries: PyTorch, Transformers, scikit-learn, pandas, numpy, seaborn, matplotlib

Install dependencies:
```bash
pip install -r requirements.txt

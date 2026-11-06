# 🏛️ Sri Lankan Case Law — Statutory Classification & XAI Engine

An end-to-end Machine Learning and Artificial Neural Network (ANN) pipeline designed to classify Sri Lankan appellate case summaries into four primary statutory domains, feature-engineered with strict group-isolated data splits to eliminate structural data leakage, 
and deployed with an Explainable AI (XAI) Streamlit dashboard.

*** Please note***: Due to size issues Models folder was uploaded separately to One drive, please find the link and apply according to the Project Repository Structure shown below.
# Models generated following 02_model_training_second.py file is runnned shown below.
# Also contains the video demonstration 
One Drive Link containing the models: https://nibm-my.sharepoint.com/personal/comscds252p-012_student_nibm_lk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fcomscds252p%2D012%5Fstudent%5Fnibm%5Flk%2FDocuments%2FANN%20Legal%20Domain%20Coursework%20Model%20Folder&ga=1 

## 📌 1. Project Overview & Key Deliverables

* **Dataset:** [`navodPeiris/sri-lanka-case-law`](https://huggingface.co/datasets/navodPeiris/sri-lanka-case-law) (Hugging Face)
* **Objective:** Multi-Class Sequence Classification across 4 statutory domains:
  1. *Civil Procedure / Property*
  2. *Penal Code / Criminal*
  3. *Statutory & Administrative*
  4. *General Appellate / Other*
* **Architecture:** 
  * **Classical Baseline:** `TfidfVectorizer` (10,000 max features, unigram/bigram) + `LinearSVC` (`class_weight='balanced'`).
  * **Deep Transformer ANN:** Fine-tuned `distilbert-base-uncased` sequence classifier trained over 4 epochs using `DataCollatorWithPadding`.
* **Interpretability & UI:** Streamlit web interface (`03_app_and_xai.py`) featuring gradient-based feature attribution and local TF-IDF coefficient highlighting.

---

## 📊 2. Final Benchmark Results

Evaluated independently on a strictly isolated test set ($N = 44$) following leak-free `GroupShuffleSplit` partitioning:

| Model Architecture | Test Macro F1-Score | Overall Accuracy | Primary Artifact |
| **Classical Baseline (TF-IDF + Linear SVM)** | `0.688267` | 68.18% (30/44) | `models/baseline_svm.pkl` |
| **Fine-Tuned Transformer ANN (DistilBERT - Epoch 4)** | **`0.769627`** | **77.27% (34/44)** | `models/legal_transformer_finetuned/` |

# DistilBERT Class-Wise Performance Breakdown
* **Civil Procedure / Property:** Precision: `0.75` | Recall: `0.60` | F1-Score: `0.67` (Support: 10)
* **General Appellate / Other:** Precision: `0.70` | Recall: `0.78` | F1-Score: `0.74` (Support: 9)
* **Penal Code / Criminal:** Precision: `0.88` | Recall: `0.88` | F1-Score: `0.88` (Support: 8)
* **Statutory & Administrative:** Precision: `0.78` | Recall: `0.82` | F1-Score: `0.80` (Support: 17)

---

## 🔒 3. Structural Data Leakage Isolation

Judicial datasets often contain multiple document entries (headnotes, full judgments, summaries, or appeal records) originating from the same underlying court case. 
Standard random splitting causes records sharing the exact same case identifier to distribute across both training and test sets, artificially inflating test scores (~0.98 Macro F1) due to metadata memorization (judges' titles, filing dates).

To enforce true out-of-group generalization:
* Data partitioning is executed via `GroupShuffleSplit` in `01_eda_and_data_prep.ipynb`.
* Partitioning is strictly conditioned on unique metadata: `group_candidates = ['case_no', 'case_title', 'citation', 'case_name', 'id', 'file_name']`.
* The focus here is to assign the data in a understnable format on the csv files
* All summaries belonging to a specific litigation proceeding are restricted exclusively to either the **Training Split (80%, N=201)** or the **Held-Out Test Split (20%, N=44)**.

---

## 📁 4. Project Repository Structure

```text
sl_legal_ai_ANN_project/
│
├── data/                                 # Cleaned, group-isolated CSV data splits
│   ├── train.csv                         # Training split (N=201)
│   ├── val.csv                           # Validation split (N=44)
│   └── test.csv                          # Unseen held-out test split (N=44)
│
├── models/                               # Serialized model weights & vectorizers
│   ├── tfidf_vectorizer.pkl              # Fitted 10,000-feature TF-IDF vectorizer
│   ├── baseline_svm.pkl                  # Trained LinearSVC baseline weights
│   ├── label_encoder.pkl                 # Target domain string-to-int LabelEncoder
│   └── legal_transformer_finetuned/      # Optimal 4th-epoch DistilBERT PyTorch weights & vocabulary
│
├── 01_eda_and_data_prep.ipynb            # Data ingestion, metadata parsing, GroupShuffleSplit
├── 02_model_training_second.py           # End-to-end model training, batch collation, & export
├── 03_app_and_xai.py                     # Streamlit web app & Explainable AI dashboard
├── 04_genai_eval_check.py                # Standalone test set evaluation & metric exporter
│
├── evaluation_summary.csv                # Automated CSV summary of model benchmark comparisons
├── .pdf # Formal PDF academic coursework report
├── requirements.txt                      # Environment dependencies
└── README.md                             # Repository documentation

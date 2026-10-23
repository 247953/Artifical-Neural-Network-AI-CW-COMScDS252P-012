import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#Get absolute path of root folder for extraction and initialisation
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

def evaluate_all_models():
    print("=== Comprehensive Model Evaluation and Benchmarking ===")

# Load test set
    test_path = os.path.join(DATA_DIR, 'test.csv')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset not found at '{test_path}'. Please run EDA splitting first!") # LETTING user know of the split and for diagnosis 
    
    test_df = pd.read_csv(test_path)
    X_test = test_df['input_text'].fillna("").astype(str)
    y_test = test_df['target_domain']
    
    print(f"Loaded Test Set: {len(X_test)} samples evaluated under leak-free Grouped Split.")
    
    # --- 1. EVALUATE CLASSICAL BASELINE (Linear SVM) ---
    print("\n--- Evaluating Baseline: TF-IDF + Linear SVM ---")
    vectorizer = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    svm_model = joblib.load(os.path.join(MODEL_DIR, 'baseline_svm.pkl'))
    
    X_test_vec = vectorizer.transform(X_test)
    y_pred_svm = svm_model.predict(X_test_vec)
    macro_f1_svm = f1_score(y_test, y_pred_svm, average='macro', zero_division=0)
    print(f"Linear SVM Test Macro F1-Score: {macro_f1_svm:.4f}")  

    # --- 2. EVALUATE FINE-TUNED TRANSFORMER ANN ---
    print("\n--- Evaluating Fine-Tuned Transformer ANN (DistilBERT) ---")
    label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    transformer_path = os.path.join(MODEL_DIR, 'legal_transformer_finetuned')
    
    tokenizer = AutoTokenizer.from_pretrained(transformer_path)
    model = AutoModelForSequenceClassification.from_pretrained(transformer_path)
    model.eval()
    
    # Batch inference on test set
    preds = []
    batch_size = 16
    texts = X_test.tolist()
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            outputs = model(**inputs)
            batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            preds.extend(batch_preds)
            
    y_pred_transformer_labels = label_encoder.inverse_transform(preds)
    macro_f1_transformer = f1_score(y_test, y_pred_transformer_labels, average='macro', zero_division=0)
    print(f"Transformer ANN Test Macro F1-Score: {macro_f1_transformer:.4f}")
    
    # --- 3. SUMMARY REPORT ---
    print("\n==================================================")
    print("         FINAL BENCHMARK SUMMARY       ")
    print("==================================================")
    summary_df = pd.DataFrame({
        "Model Architecture": [
            "Classical Baseline (TF-IDF + Linear SVM)", 
            "Fine-Tuned Transformer ANN (DistilBERT)"
        ],
        "Test Macro F1-Score": [macro_f1_svm, macro_f1_transformer]
    })
    print(summary_df.to_string(index=False))
    print("==================================================")
    
    # Save report artifact
    report_path = os.path.join(BASE_DIR, 'evaluation_summary.csv')
    summary_df.to_csv(report_path, index=False)
    print(f"\nBenchmark report table saved successfully to: {report_path}")

if __name__ == "__main__":
    evaluate_all_models()
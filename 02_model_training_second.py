import os
import pandas as pd
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding

# Get absolute path of root folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

def run_training_pipeline():
    print("=== 1. LOADING SPLIT DATASETS ===")
    train_path = os.path.join(DATA_DIR, 'train.csv')
    val_path = os.path.join(DATA_DIR, 'val.csv')
    test_path = os.path.join(DATA_DIR, 'test.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Split datasets not found in '{DATA_DIR}'. Please run your EDA notebook first!")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df['input_text'].fillna("").astype(str)
    y_train = train_df['target_domain']
    X_val = val_df['input_text'].fillna("").astype(str)
    y_val = val_df['target_domain']
    X_test = test_df['input_text'].fillna("").astype(str)
    y_test = test_df['target_domain']
    
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # --- QUESTION 2: CLASSICAL BASELINE ---
    print("\n=== 2. TRAINING CLASSICAL BASELINE (TF-IDF + Linear SVM) ===")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    svm_model = LinearSVC(class_weight='balanced', random_state=42, max_iter=2000)
    svm_model.fit(X_train_vec, y_train)
    
    y_pred_svm = svm_model.predict(X_test_vec)
    macro_f1_svm = f1_score(y_test, y_pred_svm, average='macro', zero_division=0)
    print(f"Baseline Linear SVM Test Macro F1-Score: {macro_f1_svm:.4f}")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    joblib.dump(svm_model, os.path.join(MODEL_DIR, 'baseline_svm.pkl'))
    
    # --- QUESTION 3: FINE-TUNED TRANSFORMER ANN ---
    print("\n=== 3. TRAINING LEGAL TRANSFORMER ANN (Fine-Tuning) ===")
    
    # Encode string labels to integers
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(y_train)
    val_labels = label_encoder.transform(y_val)
    test_labels = label_encoder.transform(y_test)
    num_labels = len(label_encoder.classes_)
    
    # Save label encoder for inference in the app
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    
    # Select lightweight, stable model for local fine-tuning
    model_ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    
    # Tokenization function with truncation
    def tokenize_function(batch):
        return tokenizer(batch['text'], truncation=True, max_length=512)
    
    # Build Hugging Face datasets
    train_hf = Dataset.from_dict({'text': X_train.tolist(), 'label': train_labels})
    val_hf = Dataset.from_dict({'text': X_val.tolist(), 'label': val_labels})
    
    train_tokenized = train_hf.map(tokenize_function, batched=True)
    val_tokenized = val_hf.map(tokenize_function, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels)
    
    # Define compute metrics for Trainer
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)
        return {"macro_f1": f1}
    
    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, 'legal_transformer_finetuned'),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5, # used to be 2e-5
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4, # use to be 2
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=10,
        report_to="none",
        use_cpu=True #Local CPU hardware constraint
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    
    print("Starting Transformer training (this may take a few minutes)...")
    trainer.train()
    
    # Save the final fine-tuned model ANN weights
    transformer_save_path = os.path.join(MODEL_DIR, 'legal_transformer_finetuned')
    trainer.save_model(transformer_save_path)
    tokenizer.save_pretrained(transformer_save_path)
    print(f"\nTransformer ANN model successfully trained and saved to: {transformer_save_path}")

    print("\n=== 4. EVALUATING FINE-TUNED TRANSFORMER ON TEST SET ===")
    test_hf = Dataset.from_dict({'text': X_test.tolist(), 'label': test_labels})
    test_tokenized = test_hf.map(tokenize_function, batched=True)

    # Evaluate the best loaded model (loaded via load_best_model_at_end=True) on test data
    test_results = trainer.predict(test_tokenized)
    test_preds = np.argmax(test_results.predictions, axis=-1)
    
    macro_f1_transformer = f1_score(test_labels, test_preds, average='macro', zero_division=0)
    print(f"Fine-Tuned DistilBERT Test Macro F1-Score: {macro_f1_transformer:.4f}")
    print("\nTransformer Test Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=label_encoder.classes_))

if __name__ == "__main__":
    run_training_pipeline()

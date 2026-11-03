import streamlit as st
import os
import pandas as pd
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# page configuration setup title
st.set_page_config(
    page_title="Sri Lankan Legal AI - ANN classifier and Explainable AI",
    page_icon="⚖️",
    layout="wide",
)
# Load Saved Model Artifacts (Cached for performance)
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'models')
    
    # Load Label Encoder
    label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
    
    # Load Transformer ANN (Ensure 'transformer' has both 'r's here!)
    transformer_path = os.path.join(model_dir, 'legal_transformer_finetuned')
    tokenizer = AutoTokenizer.from_pretrained(transformer_path)
    model = AutoModelForSequenceClassification.from_pretrained(transformer_path)
    model.eval()
    
    # Load Baseline Vectorizer & SVM for XAI fallback
    vectorizer = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
    svm_model = joblib.load(os.path.join(model_dir, 'baseline_svm.pkl'))
    
    return label_encoder, tokenizer, model, vectorizer, svm_model

label_encoder, tokenizer, model, vectorizer, svm_model = load_artifacts()

# App Header
st.title("⚖️ Sri Lankan Legal AI: Automated Case & Statutory Domain Classifier")
st.markdown("### Designing an Artificial Neural Networks Coursework Legal Project")
st.write("Designed for the Supreme Court Case Assignment Clerk to automate petition routing and ensure transparent Explainable AI (XAI) decision-making.")

# Sidebar Controls
st.sidebar.header("Configuration & Model Selection")
model_choice = st.sidebar.selectbox(
    "Select Architecture", 
    ["Fine-Tuned Transformer ANN (DistilBERT)", "Classical Baseline (Linear SVM)"]
)
st.sidebar.info("Dataset: navodPeiris/sri-lanka-case-law\nEvaluation Metric: Macro F1-Score\nLeakage Control: Grouped Split by Case ID")

# Main Interface Input Area
st.subheader("Enter Case Summary or Pre-Decision Judgment Text:")
default_case = (
    "The appellant was convicted under Section 300 of the Penal Code for committing culpable "
    "homicide not amounting to murder during an industrial dispute regarding wage arrears."
)
case_text = st.text_area("Case Summary / Text Input", value=default_case, height=150)

if st.button("Classify Statutory Domain", type="primary"):
    if not case_text.strip():
        st.warning("Please enter valid case text before running inference.")
    else:
        with st.spinner("Executing neural network forward pass..."):
            if "Transformer" in model_choice:
                # Transformer ANN Inference
                inputs = tokenizer(
                    case_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1).numpy()[0]
                    pred_idx = torch.argmax(logits, dim=-1).item()

                pred_domain = label_encoder.inverse_transform([pred_idx])[0]
                confidence = float(probs[pred_idx])

                st.success(f"**Predicted Statutory Domain:** {pred_domain}")
                st.metric(label="ANN Model Confidence", value=f"{confidence:.2%}")

                # Display Probability Distribution Across Domains
                st.markdown("### Class Probability Distribution")
                prob_df = pd.DataFrame({
                    "Legal Domain": label_encoder.classes_,
                    "Probability": probs,
                }).sort_values(by="Probability", ascending=False)
                st.bar_chart(prob_df.set_index("Legal Domain"))

            else:
                # Baseline SVM Inference
                vec_text = vectorizer.transform([case_text])
                pred_domain = svm_model.predict(vec_text)[0]
                st.success(f"**Predicted Statutory Domain (Baseline SVM):** {pred_domain}")

        # Explainable AI (XAI) Section
        st.markdown("---")
        st.subheader("🔍 Explainable AI (XAI) & Feature Attribution")
        st.write("Top influencing legal keywords and n-grams driving the model's routing decision:") # following keywords for reference to the user

        feature_names = vectorizer.get_feature_names_out()
        vec_array = vectorizer.transform([case_text]).toarray()[0]
        non_zero_indices = vec_array.nonzero()[0]

        if len(non_zero_indices) > 0:
            top_features = sorted(
                [(feature_names[i], vec_array[i]) for i in non_zero_indices],
                key=lambda x: x[1],
                reverse=True,
            )[:10]
            xai_df = pd.DataFrame(top_features, columns=["Keyword / N-Gram", "TF-IDF Weight"])
            st.dataframe(xai_df, use_container_width=True)
        else:
            st.info("No matching keywords found for attribution.")

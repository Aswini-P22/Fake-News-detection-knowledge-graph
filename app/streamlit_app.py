from matplotlib import colors
import spacy
import community.community_louvain as community_louvain


import streamlit as st
import pandas as pd
import joblib
import os
import re
import networkx as nx
import matplotlib.pyplot as plt


@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")
nlp = load_spacy_model()

# -----------------------------
# PATH CONFIG (TOP LEVEL ONLY)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")


@st.cache_resource
def load_models():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

model, vectorizer = load_models()


# -----------------------------
# TEXT CLEANING
# -----------------------------
def basic_clean(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# -----------------------------
# RELATION EXTRACTION
# -----------------------------
def extract_relations(text):
    doc = nlp(text)
    relations = []

    for token in doc:
        if token.pos_ == "VERB":
            subject = None
            obj = None

            for left in token.lefts:
                if left.dep_ in ("nsubj", "nsubjpass"):
                    subject = left.text

            for right in token.rights:
                if right.dep_ in ("dobj", "pobj"):
                    obj = right.text

            if subject and obj:
                relations.append((subject, token.lemma_, obj))

    return relations

# -----------------------------
# UI
# -----------------------------
st.title("📰 Fake News Detection using NLP")
st.caption("TF-IDF + ML | NER & Relation Graph (Explainable AI)")

news_text = st.text_area(
    "Enter news article text",
    height=220,
    placeholder="Paste the news article here..."
)

show_graph = st.checkbox("Show relationship graph (important entities only)")

if st.button("Analyze News"):
    if not news_text.strip():
        st.warning("Please enter some text")
        st.stop()

    # -----------------------------
    # PREDICTION
    # -----------------------------
    clean_text = basic_clean(news_text)
    X = vectorizer.transform([clean_text])
    prediction = model.predict(X)[0]

    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(X)[0].max()
    else:
        score = model.decision_function(X)[0]
        confidence = abs(score) / (abs(score) + 1)

    st.subheader("📌 Prediction")
    if prediction == 1:
        st.success("✅ TRUE NEWS")
    else:
        st.error("❌ FAKE NEWS")

    st.metric("Confidence", f"{confidence:.2%}")


    # NER
   
    doc = nlp(news_text)

    entities = [(ent.text, ent.label_) for ent in doc.ents]

    if entities:
        ner_df = pd.DataFrame(entities, columns=["Entity", "Label"])
        st.dataframe(ner_df, use_container_width=True)
    else:
        st.info("No named entities found.")

    # -----------------------------
    # RELATIONS
    # -----------------------------
    st.subheader("🔗 Relation Extraction")
    relations = extract_relations(news_text)

    if relations:
        df_rel = pd.DataFrame(relations, columns=["Subject", "Relation", "Object"])
        st.dataframe(df_rel, use_container_width=True)
    else:
        st.info("No relations extracted.")

    # -----------------------------
    # SMALL GRAPH (IMPORTANT NODES ONLY)
    # -----------------------------
    if show_graph and relations:
        st.subheader("🕸️ Entity Relationship Graph")

        BAD_NODES = {
            "he", "she", "it", "they", "them", "who",
            "that", "which", "this", "we", "i", "you"
        }

        G = nx.Graph()

        for s, r, o in relations:
            if s.lower() not in BAD_NODES and o.lower() not in BAD_NODES:
                G.add_edge(s, o, label=r)

        # keep only important nodes
        degree = dict(G.degree())
        important_nodes = [n for n, d in degree.items() if d >= 1]

        

        G = G.subgraph(important_nodes).copy()

        if G.number_of_nodes() > 20:
            top_nodes = sorted(
                degree, key=degree.get, reverse=True
            )[:20]
            G = G.subgraph(top_nodes).copy()

        fig, ax = plt.subplots(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42)

        partition = community_louvain.best_partition(G)
        colors = [partition[node] for node in G.nodes()]  

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=1800,
            node_color=colors,  
            cmap=plt.cm.Set3,    
            edge_color="gray",
            font_size=9,
            ax=ax
        )

        plt.show()

        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        st.pyplot(fig)


# import sys
# import os

# # Add project root to Python path
# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(ROOT_DIR)


# import streamlit as st
# import joblib
# import pandas as pd
# from src.utils import basic_clean, extract_entities, extract_relations

# # Load model & vectorizer
# model = joblib.load("models/best_model.pkl")
# vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# st.set_page_config(page_title="Fake News Detection", layout="wide")
# st.title("📰 Fake News Detection with Explainability")

# news_text = st.text_area("Enter News Article", height=200)

# if st.button("Analyze"):
#     if news_text.strip() == "":
#         st.warning("Please enter text")
#     else:
#         cleaned = basic_clean(news_text)
#         X = vectorizer.transform([cleaned])
#         prediction = model.predict(X)[0]

#         label = "🟢 REAL NEWS" if prediction == 1 else "🔴 FAKE NEWS"
#         st.subheader(f"Prediction: {label}")

#         # -----------------------------
#         # NER
#         # -----------------------------
#         st.subheader("🔍 Named Entities")
#         entities = extract_entities(news_text)
#         if entities:
#             st.dataframe(pd.DataFrame(entities, columns=["Entity", "Type"]))
#         else:
#             st.info("No entities found")

#         # -----------------------------
#         # Relations
#         # -----------------------------
#         st.subheader("🔗 Extracted Relations")
#         relations = extract_relations(news_text)
#         if relations:
#             st.dataframe(pd.DataFrame(relations, columns=["Subject", "Verb", "Object"]))
#         else:
#             st.info("No relations found")

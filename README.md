# 📰 Fake News Detection with Knowledge Graphs

An **end-to-end Fake News Detection system** that combines  
**Machine Learning + NLP + Knowledge Graphs + Graph Theory**,  
with an interactive **Streamlit UI** for explainable predictions.

---

## 🚀 Project Highlights

- ✅ Fake News Detection using classical ML models  
  (Logistic Regression, Random Forest, Linear SVM, Naive Bayes)
- 🧠 Advanced NLP with **Named Entity Recognition (NER)** and **Relation Extraction**
- 🕸️ Knowledge Graph construction from news articles
- 👥 **Community Detection** using the Louvain Algorithm
- 📊 **Centrality Analysis** (Degree, Betweenness, Closeness)
- 🔗 Cross-Domain Linking between news articles
- 🌐 Interactive **Streamlit Web Application**
- 📁 Clean, modular, industry-standard project structure

---

## 🧩 Problem Statement

Fake news spreads rapidly across digital platforms, influencing public opinion and decision-making.  
Traditional text classification methods lack **interpretability**.

This project goes beyond simple classification by:
- Understanding **entities** in news articles
- Extracting **relationships** between entities
- Building **knowledge graphs**
- Analyzing **important entities and communities**

---

## 🏗️ System Architecture

```text
Raw News Data
   ↓
Data Cleaning & Preprocessing
   ↓
TF-IDF Feature Extraction
   ↓
Model Training & Comparison
   ↓
Fake News Prediction
   ↓
Named Entity Recognition (NER)
   ↓
Relation Extraction
   ↓
Knowledge Graph Construction
   ↓
Community & Centrality Analysis
   ↓
Streamlit Interactive UI
```

---

## 🛠️ Technologies Used

### Programming & Libraries
- Python
- Pandas, NumPy
- Scikit-learn
- spaCy (NLP)
- NetworkX
- Matplotlib
- Streamlit

### Machine Learning Models
- Logistic Regression
- Random Forest Classifier
- Linear SVM
- Multinomial Naive Bayes

---

## 📂 Project Structure

```text
fake-news-detection-knowledge-graph/
│
├── app/
│   └── streamlit_app.py              # Streamlit UI
│
├── src/
│   ├── train_model.py                # Model training & evaluation
│   └── utils.py                      # NLP utilities (NER, relations)
│
├── notebooks/
│   ├── 1_data_exploration.ipynb
│   ├── 2_data_preprocessing.ipynb
│   ├── 3_centrality_community.ipynb
│   └── 4_cross_domain_linking.ipynb
│
├── data/
│   ├── raw_data/                     # Original datasets
│   └── preprocessed_data/            # Ignored in Git due to size
│
├── outputs/
│   └── cross_domain_links.csv
│
├── requirements.txt
├── .gitignore
└── README.md

```
---

## 📊 Model Performance (Best Model)

| Model | Accuracy | F1 Score |
|------|----------|----------|
| **Random Forest** | **99.91%** | **0.9991** |
| Linear SVM | 99.88% | 0.9988 |
| Logistic Regression | 99.72% | 0.9973 |
| Naive Bayes | 95.59% | 0.9572 |

✔ **Random Forest selected as the final model**

---

## 🕸️ Knowledge Graph & Network Analysis

- Extracted **Subject–Relation–Object** triples from news text
- Constructed graphs using **NetworkX**
- Identified:
  - Influential entities using **centrality measures**
  - Thematic communities using **Louvain clustering**
- Reduced noise by filtering low-degree nodes
- Visualized entity relationships for better interpretability

---

## 🌐 Streamlit Application

### Features
- News text input
- Fake news prediction with confidence score
- Named Entity Recognition (NER)
- Relation extraction
- Knowledge graph visualization
- Community-aware analysis

### Run the app
```bash
streamlit run app/streamlit_app.py

```

### ⚠️ Dataset Note

Due to GitHub file size limitations, the preprocessed dataset
cleaned_fakenews.csv is not included in this repository.

However, the complete preprocessing pipeline is available in:

notebooks/2_data_preprocessing.ipynb

src/utils.py

This ensures full reproducibility of the project.


### 🎯 Key Outcomes

Built an explainable AI system, not just a black-box classifier

Integrated ML + NLP + Graph Theory

Followed industry-grade project structure

Designed a scalable and interpretable fake news detection pipeline

---

## ⚙️ Model Optimization & Hyperparameter Tuning

During model development, **controlled hyperparameter tuning** was performed to improve classification performance and stability.

### Feature Engineering
- TF-IDF vectorization was applied to transform raw news text into numerical features.
- Important parameters such as vocabulary size and token filtering were adjusted to reduce noise and improve representation.

### Model Training & Optimization
- Multiple machine learning models were trained and compared.
- Logistic Regression was selected as the final model based on performance and interpretability.
- Hyperparameters such as regularization strength and solver configuration were tuned to prevent overfitting and ensure generalization.

### Model Selection Strategy
- Models were evaluated using standard classification metrics.
- The final configuration was chosen based on a balance between accuracy, robustness, and explainability.

This approach ensured the model was **optimized beyond default settings** while remaining interpretable for downstream analysis.

---

### 🔮 Future Enhancements

Transformer-based models (BERT, RoBERTa)

Temporal knowledge graph analysis

Fact-checking API integration

Multilingual fake news detection

Cloud deployment (AWS / Hugging Face Spaces)

### 👩‍💻 Author

Aswini P

Artificial Intelligence & Data Science

🔗 GitHub: https://github.com/Aswini-P22

🔗 LinkedIn: https://www.linkedin.com/in/aswini-purushothaman-2206p2006
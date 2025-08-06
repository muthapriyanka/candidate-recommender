# 🔍 Candidate Recommendation Engine

This Streamlit web app recommends the most relevant candidates for a job based on resume-job similarity, and generates a short AI summary explaining why they are a good fit.

---

## 🧠 Features

- Accepts a **job description**
- Uploads multiple **PDF resumes**
- Embeds both using **sentence-transformers** (`all-MiniLM-L6-v2`)
- Computes **cosine similarity**
- Ranks top candidates
- Uses a **free Hugging Face summarizer** (`distilbart-cnn-12-6`) to explain fit

---
## 🚀 How to Run Locally

```bash
git clone https://github.com/muthapriyanka/candidate-recommender.git
cd candidate-recommender
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

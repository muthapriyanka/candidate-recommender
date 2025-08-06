import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import PyPDF2
import os

st.set_page_config(page_title="Candidate Recommendation Engine")
st.title("🔍 Candidate Recommendation Engine")

job_description = st.text_area("📄 Paste the Job Description:")
uploaded_files = st.file_uploader("📂 Upload Resumes (PDF only)", type=["pdf"], accept_multiple_files=True)

# Extract text from uploaded PDF files
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    return " ".join(page.extract_text() for page in reader.pages if page.extract_text())

# Lazy-load SentenceTransformer (embedding model)
@st.cache_resource(show_spinner="🔄 Loading embedding model...")
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')

# Lazy-load summarizer
@st.cache_resource(show_spinner="🔄 Loading summarizer model...")
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Generate summary for fit
def generate_summary(job_desc, resume_text, summarizer):
    try:
        text = f"Job Description: {job_desc}\n\nResume: {resume_text[:1500]}"
        summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"⚠️ Could not generate summary: {str(e)}"

# Only process on button click
if st.button("Process"):
    if not job_description or not uploaded_files:
        st.warning("❗ Please enter job description and upload resumes.")
    else:
        embedder = load_embedder()
        summarizer = load_summarizer()

        job_emb = embedder.encode([job_description])
        candidates = []

        for f in uploaded_files:
            resume_text = extract_text_from_pdf(f)
            resume_emb = embedder.encode([resume_text])
            score = cosine_similarity(job_emb, resume_emb)[0][0]

            summary = generate_summary(job_description, resume_text, summarizer)

            candidates.append({
                "name": os.path.splitext(f.name)[0],
                "score": score,
                "summary": summary
            })

        top_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)[:10]

        st.subheader("🏆 Top Candidates")
        for idx, c in enumerate(top_candidates, 1):
            st.markdown(f"**{idx}. {c['name']}**")
            st.write(f"Similarity Score: `{round(c['score'] * 100)}%`")
            st.markdown("**🧠 Fit Summary:**")
            st.write(c['summary'])

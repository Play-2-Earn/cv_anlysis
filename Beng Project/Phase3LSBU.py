#This code Should work with the "LSBUPhase3.html template they everything worked last check was on 27/01/2025"
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF for PDF handling
import os
from collections import Counter

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__, template_folder='templates')

# Load Sentence Transformer model
model = SentenceTransformer('all-mpnet-base-v2')

# Helper function to extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as pdf:
        for page in pdf:
            text += page.get_text()
    return text

# Helper function to extract keywords
def extract_keywords(text):
    """Extracts keywords from text using simple tokenization."""
    words = text.split()
    common_words = set(["and", "or", "the", "a", "in", "to", "of", "for", "with", "on", "by", "as", "at", "is"])  # Stop words
    keywords = [word.strip().lower() for word in words if word.isalpha() and word.lower() not in common_words]
    return Counter(keywords)

# Home Route
@app.route('/')
def home():
    return render_template('LSBUPhase3.html')

# Route for JD and CV analysis
@app.route('/option3', methods=['GET', 'POST'])
def option3():
    if request.method == 'POST':
        jd_files = request.files.getlist('jd_files')
        cv_files = request.files.getlist('cv_files')

        if not jd_files:
            return jsonify({'error': 'Please upload at least one JD file'}), 400
        if not cv_files:
            return jsonify({'error': 'Please upload at least one CV file'}), 400

        # Extract text from JDs and CVs
        jd_texts = [extract_text_from_pdf(jd) for jd in jd_files]
        cv_texts = [extract_text_from_pdf(cv) for cv in cv_files]

        # Encode JDs and CVs using the SentenceTransformer model
        jd_embeddings = [model.encode(jd_text, convert_to_tensor=True) for jd_text in jd_texts]
        cv_embeddings = [model.encode(cv_text, convert_to_tensor=True) for cv_text in cv_texts]

        # Calculate similarity scores for each JD-CV pair
        results = []
        for jd_idx, jd_embedding in enumerate(jd_embeddings):
            jd_keywords = extract_keywords(jd_texts[jd_idx])
            for cv_idx, cv_embedding in enumerate(cv_embeddings):
                score = util.pytorch_cos_sim(jd_embedding, cv_embedding).item()
                cv_keywords = extract_keywords(cv_texts[cv_idx])
                matched_keywords = set(jd_keywords.keys()) & set(cv_keywords.keys())
                results.append({
                    "jd_name": jd_files[jd_idx].filename,
                    "cv_name": cv_files[cv_idx].filename,
                    "similarity_score": round(score, 4),
                    "matched_keywords": list(matched_keywords),
                    "total_matches": len(matched_keywords)
                })

        return jsonify({'results': results})

    return render_template('phase3HTML.html')

if __name__ == "__main__":
    app.run(debug=True, port=5001)

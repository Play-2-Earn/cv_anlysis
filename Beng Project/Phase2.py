import re
import fitz  # PyMuPDF for PDF handling
import docx  # for handling DOCX files
import os
from fuzzywuzzy import fuzz  # for fuzzy matching
from gensim.scripts.glove2word2vec import glove2word2vec  # For GloVe to Word2Vec conversion
from pymongo import MongoClient
from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# Reuse the same MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['cv_analysis2_db']    # same DB name as in your main code
cv_collection = db['cvs2']        # same collection name as in your main code

# Folder to store uploaded JDs (create if not existing)
JD_UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploaded_jds')
os.makedirs(JD_UPLOAD_FOLDER, exist_ok=True)

# --- Helper functions for text extraction (same logic you used for CVs) ---
def extract_text_from_pdf(file_stream):
    """
    Extract text from a PDF file stream.
    """
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    full_text = []
    for page in doc:
        full_text.append(page.get_text())
    return "\n".join(full_text)

def extract_text_from_docx(file_stream):
    """
    Extract text from a DOCX file stream.
    """
    docx_obj = docx.Document(file_stream)
    full_text = []
    for para in docx_obj.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

def preprocess_text(text):
    """
    Simple text preprocessor: remove special chars, convert to lowercase.
    """
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower()


# --- Route to upload JD and then match CVs ---
@app.route('/upload_jd', methods=['POST'])
def upload_jd():
    """
    1) Receive the JD as a PDF or DOCX.
    2) Extract text from JD.
    3) Extract basic skill keywords (naive approach).
    4) Compare JD skills to each CV's skills in the database.
    5) Return sorted list of best matching CVs.
    """
    jd_file = request.files.get('jd_file')
    if not jd_file:
        return jsonify({"error": "No JD file uploaded."}), 400

    # Save JD file locally
    jd_path = os.path.join(JD_UPLOAD_FOLDER, jd_file.filename)
    jd_file.save(jd_path)

    # Determine file extension
    extension = jd_file.filename.split('.')[-1].lower()

    # Extract JD text
    try:
        if extension == 'pdf':
            with open(jd_path, 'rb') as f:
                jd_text = extract_text_from_pdf(f)
        elif extension == 'docx':
            with open(jd_path, 'rb') as f:
                jd_text = extract_text_from_docx(f)
        else:
            return jsonify({"error": "Unsupported JD file format. Please upload PDF or DOCX."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Preprocess the JD text
    jd_text_processed = preprocess_text(jd_text)

    # --- Extract naive JD skills by splitting on spaces, then filter a known set of skill keywords ---
    # For demonstration, let's define a small set of potential skill keywords.
    # In practice, you'd have a more robust approach or re-use your ESCO extraction method.
    possible_skills = [
        "python", "machine learning", "java", "javascript", "tensorflow",
        "react", "c++", "aws", "docker", "sql", "r", "data", "analysis"
    ]

    jd_skills_found = []
    for skill in possible_skills:
        if skill in jd_text_processed:
            jd_skills_found.append(skill)

    # Now let's compare to each CV in the DB
    all_cvs = list(cv_collection.find({}))

    # We'll do a simple scoring approach: # of matched skills between JD and CV
    # match_score = intersection size / total JD skills
    # Store results in a list so we can sort and return
    cv_matches = []
    for cv in all_cvs:
        cv_skills = cv.get("skills", [])
        # Convert each skill in the CV to lowercase
        cv_skills_lower = [s.lower() for s in cv_skills]

        # Intersection
        matched_skills = set(jd_skills_found).intersection(set(cv_skills_lower))
        total_jd_skills = len(jd_skills_found)
        if total_jd_skills == 0:
            score = 0
        else:
            score = (len(matched_skills) / total_jd_skills) * 100  # percentage

        cv_matches.append({
            "cv_id": str(cv.get("_id")),  # Convert ObjectId to string
            "contact_info": cv.get("contact_info", {}),
            "education": cv.get("education", ""),
            "experience": cv.get("experience", ""),
            "skills": cv.get("skills", []),
            "years_of_experience": cv.get("years_of_experience", ""),
            "filename": cv.get("filename", ""),
            "matched_skills": list(matched_skills),
            "match_score": score
        })

    # Sort the CVs by descending match_score
    cv_matches_sorted = sorted(cv_matches, key=lambda x: x["match_score"], reverse=True)

    return jsonify({
        "jd_skills_found": jd_skills_found,
        "cv_matches": cv_matches_sorted
    })


@app.route('/')
def index():
    """
    Simple route to serve a minimal HTML form or instruction.
    """
    return render_template('LSBUPhase2.html')  # We'll create JDMatching.html KEEP THIS AS LSBUPhase2.html CHECK THE "template" folder


if __name__ == "__main__":
    app.run(debug=True, port=5002)

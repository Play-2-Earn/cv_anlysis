'''
# Code Below works for only Hugging face platform model transformers 
#paraphrase-MiniLM-L6-v2   roberta-large-nli-stsb-mean-tokens
#multi-qa-mpnet-base-dot-v1   paraphrase-mpnet-base-v2   all-roberta-large-v1    all-mpnet-base-v2

import fitz  # PyMuPDF for PDF handling
from sentence_transformers import SentenceTransformer, util
import os
from tkinter import Tk, filedialog
import pandas as pd
import tensorflow_hub as hub 

# Load Sentence Transformer model
model = SentenceTransformer('all-mpnet-base-v2')

# Helper function to extract text from PDF
def extract_text_from_pdf(filepath):
    """Extracts text from a PDF file without any preprocessing."""
    text = ""
    with fitz.open(filepath) as pdf:
        for page in pdf:
            text += page.get_text("text")  # Extracts raw text
    return text

# File selection dialog
def select_files(title, filetypes):
    """Opens a file dialog to select files."""
    Tk().withdraw()  # Hide the main Tkinter window
    files = filedialog.askopenfilenames(title=title, filetypes=filetypes)
    return files

# Main script
def main():
    print("Welcome to the Candidate-JD Matching System!")
 
    # Select JD PDFs
    print("Please select the Job Description (JD) PDFs.")
    jd_filepaths = select_files("Select Job Description PDFs", [("PDF files", "*.pdf")])
    if not jd_filepaths:
        print("No JD files selected. Exiting.")
        return

    jd_texts = [extract_text_from_pdf(filepath) for filepath in jd_filepaths]

    # Select CV PDFs
    print("Please select Candidate CV PDFs.")
    cv_filepaths = select_files("Select Candidate CV PDFs", [("PDF files", "*.pdf")])
    if not cv_filepaths:
        print("No CV files selected. Exiting.")
        return

    cv_texts = [extract_text_from_pdf(filepath) for filepath in cv_filepaths]

    # Encode JDs and CVs
    jd_embeddings = [model.encode(jd_text, convert_to_tensor=True) for jd_text in jd_texts]
    cv_embeddings = [model.encode(cv_text, convert_to_tensor=True) for cv_text in cv_texts]

    # Calculate similarity scores
    scores = []
    for jd_embedding in jd_embeddings:
        jd_scores = [util.pytorch_cos_sim(jd_embedding, cv_embedding).item() for cv_embedding in cv_embeddings]
        scores.append(jd_scores)

    # Prepare results for Excel
    jd_row_names = [os.path.basename(jd) for jd in jd_filepaths]
    cv_column_names = [os.path.basename(cv) for cv in cv_filepaths]
    results_df = pd.DataFrame(scores, index=jd_row_names, columns=cv_column_names) * 100
    results_df = results_df.round(2)  # Convert to percentage and round to 2 decimal places

    # Transpose the DataFrame
    results_df = results_df.T

    # Save results to Excel
    output_filename = "JD_CV_Similarity_Scores_USE.xlsx"
    results_df.to_excel(output_filename, index=True, header=True)
    print(f"Similarity scores have been saved to {output_filename}")

if __name__ == "__main__":
    main()
    

#---------------------------------------------------------------------Universal sentence encoder code

import fitz  # PyMuPDF for PDF handling
from sentence_transformers import util  # Keeping for utility functions
import os
from tkinter import Tk, filedialog
import pandas as pd
import tensorflow_hub as hub  # For loading Universal Sentence Encoder

# Load Universal Sentence Encoder model
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")  # Load USE model

# Helper function to extract text from PDF
def extract_text_from_pdf(filepath):
    """Extracts text from a PDF file."""
    text = ""
    with fitz.open(filepath) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

# File selection dialog
def select_files(title, filetypes):
    """Opens a file dialog to select files."""
    Tk().withdraw()  # Hide the main Tkinter window
    files = filedialog.askopenfilenames(title=title, filetypes=filetypes)
    return files

# Helper function to encode text using USE
def encode_text_with_use(text):
    """Encodes text using the Universal Sentence Encoder."""
    return model([text]).numpy()[0]  # Extract the first array for the embedding

# Main script
def main():
    print("Welcome to the Candidate-JD Matching System!")

    # Select JD PDFs
    print("Please select the Job Description (JD) PDFs.")
    jd_filepaths = select_files("Select Job Description PDFs", [("PDF files", "*.pdf")])
    if not jd_filepaths:
        print("No JD files selected. Exiting.")
        return

    jd_texts = [extract_text_from_pdf(filepath) for filepath in jd_filepaths]

    # Select CV PDFs
    print("Please select Candidate CV PDFs.")
    cv_filepaths = select_files("Select Candidate CV PDFs", [("PDF files", "*.pdf")])
    if not cv_filepaths:
        print("No CV files selected. Exiting.")
        return

    cv_texts = [extract_text_from_pdf(filepath) for filepath in cv_filepaths]

    # Encode JDs and CVs
    jd_embeddings = [encode_text_with_use(jd_text) for jd_text in jd_texts]
    cv_embeddings = [encode_text_with_use(cv_text) for cv_text in cv_texts]

    # Calculate similarity scores
    scores = []
    for jd_embedding in jd_embeddings:
        jd_scores = [util.cos_sim(jd_embedding, cv_embedding).numpy().item() for cv_embedding in cv_embeddings]
        scores.append(jd_scores)

    # Prepare results for Excel
    jd_row_names = [os.path.basename(jd) for jd in jd_filepaths]
    cv_column_names = [os.path.basename(cv) for cv in cv_filepaths]
    results_df = pd.DataFrame(scores, index=jd_row_names, columns=cv_column_names) * 100
    results_df = results_df.round(2)  # Convert to percentage and round to 2 decimal places

    # Transpose the DataFrame
    results_df = results_df.T

    # Save results to Excel
    output_filename = "1_JD_CV_Similarity_Scores_USE.xlsx"
    results_df.to_excel(output_filename, index=True, header=True)
    print(f"Similarity scores have been saved to {output_filename}")

if __name__ == "__main__":
    main()

'''

import fitz  # PyMuPDF for PDF handling
from sentence_transformers import util  # Keeping for utility functions
import os
from tkinter import Tk, filedialog
import pandas as pd
import tensorflow_hub as hub  # For loading Universal Sentence Encoder

# Set a custom TensorFlow Hub cache directory
os.environ["TFHUB_CACHE_DIR"] = r"C:\Users\zeiad\tfhub_cache"

# Load Universal Sentence Encoder model
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")  # Load USE model

# Helper function to extract text from PDF
def extract_text_from_pdf(filepath):
    """Extracts text from a PDF file."""
    text = ""
    with fitz.open(filepath) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

# File selection dialog
def select_files(title, filetypes):
    """Opens a file dialog to select files."""
    Tk().withdraw()  # Hide the main Tkinter window
    files = filedialog.askopenfilenames(title=title, filetypes=filetypes)
    return files

# Helper function to encode text using USE
def encode_text_with_use(text):
    """Encodes text using the Universal Sentence Encoder."""
    return model([text]).numpy()[0]  # Extract the first array for the embedding

# Main script
def main():
    print("Welcome to the Candidate-JD Matching System!")

    # Select JD PDFs
    print("Please select the Job Description (JD) PDFs.")
    jd_filepaths = select_files("Select Job Description PDFs", [("PDF files", "*.pdf")])
    if not jd_filepaths:
        print("No JD files selected. Exiting.")
        return

    jd_texts = [extract_text_from_pdf(filepath) for filepath in jd_filepaths]

    # Select CV PDFs
    print("Please select Candidate CV PDFs.")
    cv_filepaths = select_files("Select Candidate CV PDFs", [("PDF files", "*.pdf")])
    if not cv_filepaths:
        print("No CV files selected. Exiting.")
        return

    cv_texts = [extract_text_from_pdf(filepath) for filepath in cv_filepaths]

    # Encode JDs and CVs
    jd_embeddings = [encode_text_with_use(jd_text) for jd_text in jd_texts]
    cv_embeddings = [encode_text_with_use(cv_text) for cv_text in cv_texts]

    # Calculate similarity scores
    scores = []
    for jd_embedding in jd_embeddings:
        jd_scores = [util.cos_sim(jd_embedding, cv_embedding).numpy().item() for cv_embedding in cv_embeddings]
        scores.append(jd_scores)

    # Prepare results for Excel
    jd_row_names = [os.path.basename(jd) for jd in jd_filepaths]
    cv_column_names = [os.path.basename(cv) for cv in cv_filepaths]
    results_df = pd.DataFrame(scores, index=jd_row_names, columns=cv_column_names) * 100
    results_df = results_df.round(2)  # Convert to percentage and round to 2 decimal places

    # Transpose the DataFrame
    results_df = results_df.T

    # Save results to Excel
    output_filename = "1_JD_CV_Similarity_Scores_USE.xlsx"
    results_df.to_excel(output_filename, index=True, header=True)
    print(f"Similarity scores have been saved to {output_filename}")

if __name__ == "__main__":
    main()

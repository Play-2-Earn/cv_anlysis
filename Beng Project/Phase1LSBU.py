#This code Should work with the "LSBUPhase1.html template they everything worked last check was on 27/01/2025"
import re
import fitz  # PyMuPDF for PDF handling
import docx  # for handling DOCX files
import os
import pandas as pd
from PIL import Image
import pytesseract
import io
import nltk
from sentence_transformers import SentenceTransformer, util  # for BERT models
from PyQt5.QtWidgets import QApplication, QFileDialog  # PyQt5 imports
import sys
from fuzzywuzzy import fuzz  # for fuzzy matching
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec  # For GloVe to Word2Vec conversion
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pymongo import MongoClient
from flask import Flask, render_template, request, jsonify, send_file, abort, url_for, redirect
import hashlib
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import schedule
import time
import json
import requests  # Added for ESCO API integration
import os
print("Current Working Directory:", os.getcwd())


# Global variable to store dynamically generated skills from ESCO API
extracted_skills_ESCO = []

app = Flask(__name__)

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['cv_analysis2_db']  # Database name
cv_collection = db['cvs2']  # Collection name

# Directory to save uploaded files
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploaded_files')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the folder exists

# Preprocessing: clean the text (lowercase, remove special characters)
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower()

# Function to save CV data into MongoDB in an organized way
def save_cv_to_mongodb(filename, structured_data):
    print("Saving CV to MongoDB:", structured_data)  # Debugging print
    cv_hash = hashlib.sha256(structured_data['cv_text'].encode()).hexdigest()
    existing_cv = cv_collection.find_one({"cv_hash": cv_hash})

    if existing_cv:
        print("This CV already exists in the database.")
    else:
        structured_data['cv_hash'] = cv_hash
        structured_data['filename'] = filename
        cv_collection.insert_one(structured_data)
        print("CV saved to the database.")

# Function to extract text from PDFs using PyMuPDF
def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        return " ".join([page.get_text() for page in doc])
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        raise

# Function to extract text from DOCX files
def extract_text_from_docx(docx_file):
    try:
        doc = docx.Document(docx_file)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error extracting text from DOCX: {str(e)}")
        raise

# Function to extract contact information
def extract_contact_info(cv_text):
    email = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0.-]+\.[A-Z|a-z]{2,}\b', cv_text)
    phone = re.findall(r'\b\d{10,13}\b', cv_text)  # A basic pattern for phone numbers
    return {
        "email": email[0] if email else "Not Found",
        "phone": phone[0] if phone else "Not Found"
    }

# Function to extract education details (simplified regex logic)
def extract_education(cv_text):
    education_keywords = ["Bachelor", "Master", "PhD", "BSc", "MSc", "MBA", "BA", "BS", "MS", "University", "College", "Degree"]
    education_section = ""
    for keyword in education_keywords:
        match = re.search(rf'{keyword}.*\n', cv_text, re.IGNORECASE)
        if match:
            education_section += match.group() + '\n'
    return education_section.strip() if education_section else "Not Found"

# Function to extract work experience (simplified logic)
def extract_experience(cv_text):
    experience_keywords = ["Experience", "Work History", "Employment", "Projects", "Responsibilities", "Duties"]
    experience_section = ""
    for keyword in experience_keywords:
        match = re.search(rf'{keyword}.*\n(.*\n)+', cv_text, re.IGNORECASE)
        if match:
            experience_section += match.group() + '\n'
    return experience_section.strip() if experience_section else "Not Found"

# Function to retrieve skills from ESCO API
def get_skills_from_esco(job_title):
    esco_api_url = "https://ec.europa.eu/esco/api/search"
    try:
        response = requests.get(
            esco_api_url,
            params={
                'text': job_title,
                'type': 'occupation',
                'limit': 1  # Get the best match
            }
        )
        if response.status_code == 200:
            esco_data = response.json()
            if esco_data and esco_data['results']:
                occupation_uri = esco_data['results'][0]['uri']

                # Fetch skills based on occupation URI
                skills_response = requests.get(f"https://ec.europa.eu/esco/api/resource/occupation/{occupation_uri}/skills")
                if skills_response.status_code == 200:
                    skills_data = skills_response.json()
                    return [skill['title'] for skill in skills_data]
                else:
                    print(f"Error retrieving skills from ESCO: {skills_response.status_code}")
                    return []
            else:
                print("No matching occupation found")
                return []
        else:
            print(f"Error retrieving occupation from ESCO: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error calling ESCO API: {str(e)}")
        return []

# Function to extract skills from the CV
def extract_skills(cv_text):
    global extracted_skills_ESCO  # Access the global skills list
    skills = []
    predefined_skills = ["python", "machine learning", "java", "javascript", "tensorflow", "react", "c++", "aws", "docker"]

    # Combine predefined skills with dynamically fetched skills
    all_skills = predefined_skills + extracted_skills_ESCO

    for skill in all_skills:
        if skill.lower() in cv_text.lower():
            skills.append(skill)
    
    return skills if skills else "Not Found"


# Function to organize CV data
def organize_cv_data(cv_text):
    return {
        "contact_info": extract_contact_info(cv_text),
        "education": extract_education(cv_text),
        "experience": extract_experience(cv_text),
        "skills": extract_skills(cv_text),
        "cv_text": cv_text  # Full CV text is also stored
    }

# Function to calculate weighted skill score
def calculate_skill_score(user_skills, game_skills):
    total_score = 0
    for skill, weight in game_skills.items():
        if skill in user_skills:
            total_score += weight * 100  # Scale up the weights to percentages
    return total_score



# Home route
@app.route('/')
def home():
    return render_template('LSBUPhase1.html') #LSBUPhase1.html


#-------------------------------------------------------------------------------orginal Upload CV method
# Route to upload CV
@app.route('/upload_cv', methods=['POST'])
def upload_cv():
    uploaded_file = request.files['file']

    if uploaded_file:
        try:
            # Save the file temporarily
            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
            uploaded_file.save(file_path)

            file_extension = uploaded_file.filename.split('.')[-1].lower()
            print(f"Received file: {uploaded_file.filename}")  # Debugging print

            if file_extension == 'pdf':
                with open(file_path, 'rb') as pdf_file:
                    cv_text = extract_text_from_pdf(pdf_file)
                print("PDF processed successfully")
            elif file_extension == 'docx':
                with open(file_path, 'rb') as docx_file:
                    cv_text = extract_text_from_docx(docx_file)
                print("DOCX processed successfully")
            else:
                print("Unsupported file format")
                return jsonify({'error': 'Unsupported file format'}), 400

            processed_cv_text = preprocess_text(cv_text)
            structured_data = organize_cv_data(processed_cv_text)

            print("Processed CV data:", structured_data)  # Debugging print

            pdf_url = url_for('view_pdf', filename=uploaded_file.filename)
            print(f"PDF URL: {pdf_url}")  # Debugging print

            return jsonify({
                'skills': structured_data['skills'],
                'cv_text': cv_text,
                'pdf_url': pdf_url,  # Return the PDF URL
                'contact_info': structured_data['contact_info']  # Include contact information
            })

        except Exception as e:
            print(f"Error processing the file: {str(e)}")  # Print error details for debugging
            return jsonify({'error': f'Error processing the file: {str(e)}'}), 500
    else:
        print("No file uploaded")  # Debugging print
        return jsonify({'error': 'No file uploaded'}), 400


@app.route('/search_jobs', methods=['GET'])
def search_jobs():
    job_title = request.args.get('job_title')
    esco_api_url = "https://ec.europa.eu/esco/api/search"
    
    try:
        # Call ESCO API to search for job titles
        response = requests.get(
            esco_api_url,
            params={
                'text': job_title,
                'type': 'occupation',
                'limit': 5  # Return top 5 matches
            }
        )
        
        # Debugging: Print the API response or status code
        if response.status_code == 200:
            esco_data = response.json()
            print("ESCO Job search response:", esco_data)  # Debugging print

            # Ensure we access the correct path to 'results'
            if '_embedded' in esco_data and 'results' in esco_data['_embedded']:
                jobs = [result['title'] for result in esco_data['_embedded']['results']]
                return jsonify({'jobs': jobs})
            else:
                return jsonify({'error': 'No results found in ESCO API'}), 500
        else:
            print(f"Error in job search, status code: {response.status_code}")  # Debugging print
            return jsonify({'error': 'Error retrieving jobs from ESCO API'}), 500
    except Exception as e:
        print(f"Error calling ESCO API: {str(e)}")  # Debugging print
        return jsonify({'error': str(e)}), 500




#-----------------------------------------------------------------------------------------Practice 1 original this has the raw JSON of skills 
"""
@app.route('/get_job_skills', methods=['GET'])
def get_job_skills():
    job_title = request.args.get('job_title')
    
    # Step 1: Search for the job/occupation using searchGet API
    search_url = f"https://ec.europa.eu/esco/api/search?text={job_title}&type=occupation&limit=5"
    
    search_response = requests.get(search_url)
    
    if search_response.status_code == 200:
        search_results = search_response.json()
        
        if '_embedded' in search_results and 'results' in search_results['_embedded']:
            # Get the occupation URI
            occupation = search_results['_embedded']['results'][0]
            occupation_uri = occupation['uri']
            
            # Debugging: Print the occupation URI
            print(f"Occupation URI: {occupation_uri}")
            
            # Step 2: Use resourceSkillGet API to fetch skills using the occupation URI
            skill_api_url = "https://ec.europa.eu/esco/api/resource/skill"
            
            skill_response = requests.get(
                skill_api_url,
                params={'uri': occupation_uri}  # Correct parameter usage
            )
            
            # Debugging: Print the skill URL being called
            print(f"Fetching skills from: {skill_api_url}?uri={occupation_uri}")
            
            # Log the response status and content for debugging
            print(f"Skills Response Status Code: {skill_response.status_code}")
            print(f"Skills Response Content: {skill_response.content}")
            
            if skill_response.status_code == 200:
                skills_data = skill_response.json()
                return jsonify(skills_data)
            else:
                print(f"Error fetching skills: {skill_response.status_code}")
                return jsonify({'error': 'Error fetching job skills'}), 500
        else:
            return jsonify({'error': 'No occupations found'}), 404
    else:
        return jsonify({'error': 'Error fetching job suggestions'}), 500
"""



@app.route('/get_job_skills', methods=['GET'])
def get_job_skills():
    global extracted_skills_ESCO  # Declare that we are using the global variable
    job_title = request.args.get('job_title')
    
    # Step 1: Search for the job/occupation using searchGet API
    search_url = f"https://ec.europa.eu/esco/api/search?text={job_title}&type=occupation&limit=5"
    search_response = requests.get(search_url)
    
    if search_response.status_code == 200:
        search_results = search_response.json()
        if '_embedded' in search_results and 'results' in search_results['_embedded']:
            occupation = search_results['_embedded']['results'][0]
            occupation_uri = occupation['uri']
            
            # Debugging print if needed
            print(f"Occupation URI: {occupation_uri}")
            
            skill_api_url = "https://ec.europa.eu/esco/api/resource/skill"
            skill_response = requests.get(skill_api_url, params={'uri': occupation_uri})
            
            # Debugging print if needed
            print(f"Fetching skills from: {skill_api_url}?uri={occupation_uri}")
            print(f"Skills Response Status Code: {skill_response.status_code}")
            
            if skill_response.status_code == 200:
                # Extract and deduplicate titles
                titles = re.findall(r'"title":\s*"([^"]+)"', skill_response.content.decode('utf-8'))
                extracted_skills_ESCO = list(set(titles))  # Remove duplicates and store as a list

                print("\nExtracted Titles:", extracted_skills_ESCO)  # Output deduplicated titles to the terminal

                skills_data = skill_response.json()
                return jsonify({
                    'skills_data': skills_data,  # Include the raw JSON if necessary
                    'extracted_titles': extracted_skills_ESCO   # Titles extracted for skill comparison
                })
            else:
                print(f"Error fetching skills: {skill_response.status_code}")
                return jsonify({'error': 'Error fetching job skills'}), 500
        else:
            return jsonify({'error': 'No occupations found'}), 404
    else:
        return jsonify({'error': 'Error fetching job suggestions'}), 500



# Route to view PDF
@app.route('/view_pdf/<filename>')
def view_pdf(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        try:
            return send_file(file_path)
        except Exception as e:
            print(f"Error serving PDF file: {str(e)}")  # Debugging print
            return jsonify({'error': str(e)})
    else:
        print(f"File not found: {filename}")  # Debugging print
        return jsonify({'error': f'File {filename} not found'}), 404

# Route to test MongoDB connection
@app.route('/test_mongodb')
def test_mongodb():
    try:
        cv_count = cv_collection.count_documents({})
        return f'MongoDB connected. {cv_count} CVs found in the database.'
    except Exception as e:
        return f'Error connecting to MongoDB: {str(e)}'

# Route to process skills and save them
@app.route('/process_skills', methods=['POST'])
def process_skills():
    skills = request.form.getlist('skills')
    cv_text = request.form['cv_text']
    years_of_experience = request.form['experience']

    contact_info = {
        'email': request.form['email'],
        'phone': request.form['phone']
    }

    structured_data = organize_cv_data(cv_text)
    structured_data['skills'] = skills
    structured_data['years_of_experience'] = years_of_experience
    structured_data['contact_info'] = contact_info

    save_cv_to_mongodb("user_uploaded_cv", structured_data)
    return jsonify({'message': 'Skills, experience, and contact information saved successfully'})

# Route to display games and calculate skill match
@app.route('/games')
def games():
    games_data = [
        {
            "name": "Website Game",
            "skills_required": {"C++": 0.5, "Java": 0.5},
            "image": "https://www.orientsoftware.com/Themes/Content/Images/blog/2021-11-26/game-programming-languages.jpg",
            "required_score": 75
        },
        {
            "name": "Ai Survey",
            "skills_required": {"machine learning": 0.25, "python": 0.50, "tensorflow": 0.25},
            "image": "https://www.questionpro.com/blog/wp-content/uploads/2024/02/AI-Survey.jpg",
            "required_score": 75
        },
        {
            "name": "Cardano Warriors",
            "skills_required": {"React": 0.5, "JavaScript": 0.5},
            "image": "https://play2earn.net/wp-content/uploads/2022/01/cardano_warriors_artwork.png",
            "required_score": 75
        },
        {
            "name": "Mars4",
            "skills_required": {"docker": 0.5, "aws": 0.5},
            "image": "https://cdn1.epicgames.com/spt-assets/8715a708a6b4413489cb7c0a50fb8474/mars4-rfe0a.png",
            "required_score": 75
        }
    ]

    user_cv = cv_collection.find_one(sort=[("_id", -1)])  # Get the latest uploaded CV
    user_skills = [skill.lower() for skill in user_cv.get("skills", [])] if user_cv else []

    response_data = []
    for game in games_data:
        game_skills = list(game['skills_required'].keys())

        # Calculate skills the user has and is missing
        matched_skills = [skill for skill in game_skills if skill in user_skills]
        missing_skills = [skill for skill in game_skills if skill not in user_skills]

        # Calculate user's score based on the matched skills
        user_score = calculate_skill_score(matched_skills, game['skills_required'])
        can_play = user_score >= game['required_score']
        game_status = "You can play this game!" if can_play else f"You need {game['required_score'] - user_score}% more to unlock this game."

        response_data.append({
            "name": game['name'],
            "image": game['image'],
            "status": game_status,
            "user_score": user_score,
            "required_score": game['required_score'],
            "matched_skills": matched_skills,
            "missing_skills": missing_skills
        })

    return jsonify({"games": response_data, "user_skills": user_skills})

# Route to get game details
@app.route('/game/<game_name>')
def game_detail(game_name):
    game_details = {
        "Website Game": {
            "skills_required": ["C++", "Java"],
            "description": "A web development game that involves coding in C++ and Java.",
            "image": "https://cms-assets.themuse.com/media/lead/01212022-1047259374-coding-classes_scanrail.jpg"
        },
        "Ai Survey": {
            "skills_required": ["machine learning", "python", "tensorflow"],
            "description": "An AI-based survey tool requiring Python, TensorFlow, and machine learning expertise.",
            "image": "https://www.questionpro.com/blog/wp-content/uploads/2024/02/AI-Survey.jpg"
        },
        "Cardano Warriors": {
            "skills_required": ["React", "JavaScript"],
            "description": "A card-based game where you need React and JavaScript skills to develop the front end.",
            "image": "https://play2earn.net/wp-content/uploads/2022/01/cardano_warriors_artwork.png"
        },
        "Mars4": {
            "skills_required": ["docker", "aws"],
            "description": "A space simulation game where Docker and AWS are required to handle cloud operations.",
            "image": "https://cdn1.epicgames.com/spt-assets/8715a708a6b4413489cb7c0a50fb8474/mars4-rfe0a.png"
        }
    }

    if game_name in game_details:
        game_info = game_details[game_name]
        game_info["matched_skills"] = []  # Add matched skills here if available
        game_info["missing_skills"] = []  # Add missing skills here if available
        return jsonify(game_info)
    else:
        return jsonify({"error": "Game not found"}), 404

# Email-related functions for reminders
def get_user_emails():
    user_emails = []
    for cv in cv_collection.find({}, {'contact_info.email': 1}):
        email = cv.get('contact_info', {}).get('email', None)
        if email:
            user_emails.append(email)
    return user_emails

def send_email(recipient_email, subject, message):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    smtp_user = "peterpan10133@gmail.com"  # Replace with your email
    smtp_password = "avaoafezicbxljjj"  # Replace with your app password or SMTP password

    msg = MIMEMultipart()
    msg['From'] = smtp_user
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
        server.quit()
        print(f"Email sent to {recipient_email}")
    except Exception as e:
        print(f"Failed to send email to {recipient_email}: {e}")

# Function to create the reminder message
def create_email_message():
    link_to_games = "http://127.0.0.1:5001/games"
    message = f"""
    Hello,

    This is a reminder that you have new games unlocked! Click the link below to view the available games:

    {link_to_games}

    Best regards,
    Your Game Platform Team
    """
    return message

# Function to send reminders to all users
def send_reminders():
    user_emails = get_user_emails()
    message = create_email_message()
    subject = "New Games Unlocked - Check them out!"

    for email in user_emails:
        send_email(email, subject, message)

# Schedule email reminders every Monday at 9:00 AM
schedule.every().wednesday.at("15:00").do(send_reminders)

print("Scheduler is running...")

# Run the scheduler
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(60)

# Start the scheduler in a separate thread if needed, or directly in __main__
if __name__ == "__main__":
    import threading
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.start()
    app.run(debug=True, port=5001)
    
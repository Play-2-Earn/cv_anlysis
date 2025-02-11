Project Name: AI CV Analysis System 

📖 Overview
This project is a three-phase AI-driven CV analysis system that automates skill extraction, job description (JD) matching, and candidate ranking. It utilizes NLP, AI embeddings (BERT-based models), fuzzy matching, and MongoDB for structured storage.

📌 Features



Phase 1: Skill Matching & Survey/Quiz Access
🔹 Preprocess CVs: Supports PDF and DOCX (using PyMuPDF & python-docx)
🔹 Extract Information: Uses regex to extract emails, phone numbers, education, experience, and skills
🔹 AI Skill Detection: Integrates ESCO API to extract skills dynamically from over 3,000 job titles
🔹 Structured Data Storage: Saves processed data to MongoDB
🔹 Flask-Based UI: Allows users to upload CVs and get matched with quizzes/surveys based on skills
📂 Files for Phase 1:
Phase1LSBU.py - Backend processing & AI integration
LSBUPhase1.html - Web-based UI


Phase 2: AI-Powered JD Matching and CV Scoring
🔹 Preprocess JDs: Supports PDF/DOCX extraction and text cleaning
🔹 Skill Extraction from JD: Uses a predefined skills list + NLP detection (ESCO API for future scalability)
🔹 AI-Based CV Matching: Matches CVs against JD skills and ranks candidates based on AI scoring
🔹 Flask Web Interface: Allows users to upload JDs and view ranked CVs
📂 Files for Phase 2:
Phase2LSBU.py - JD processing & AI matching
LSBUPhase2.html - Web interface


Phase 3: AI-Powered JD-CV Scoring & Ranking
🔹 Preprocess CVs & JDs: Extracts text from PDFs
🔹 AI Embedding Model: Uses Sentence-BERT (all-mpnet-base-v2) to compute semantic similarity
🔹 Cosine Similarity Matching: Assigns similarity scores (0-1 scale) to rank candidates
🔹 Keyword Matching: Extracts keywords from CVs & JDs to provide insights into matches
🔹 Heatmap Visualization: Displays matching quality (green = high match, red = poor match)
📂 Files for Phase 3:
Phase3LSBU.py - AI-driven JD-CV scoring
LSBUPhase3.html - Web-based ranking interface

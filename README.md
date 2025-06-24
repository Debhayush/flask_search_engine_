# 🔍 Flask-Based Coding Problem Search Engine

This is a Flask web application that allows users to search coding problems based on TF-IDF (Term Frequency–Inverse Document Frequency) similarity to a query string. It matches user queries to relevant LeetCode-style problems.

---

## 🚀 How to Run the App

Make sure you have all dependencies installed (e.g., Flask), then run:

```bash
python app.py

flask_search_app/
│
├── app.py                  # Main Flask server to handle user input and display results
│
├── data/                   # Core dataset used for the search engine
│   ├── problemdata/        # Text files with problem titles and statements (used for TF-IDF generation)
│   ├── problemtitles/      # Collection of all problem titles
│   ├── problemurls/        # Collection of all problem URLs
│   └── problems/           # Full problem text files (excluding titles)
│
├── LeetcodeScraped/
│   ├── tfidfgen.py         # Script to generate tfidf.txt, magnitude.txt, keyword.txt, and idf.txt
│   └── test.py             # Script to test the TF-IDF algorithm using 5 sample files and a sample query
│
├── tfidf.txt               # Final TF-IDF vectors for each problem
├── magnitude.txt           # Precomputed magnitudes for cosine similarity
├── keyword.txt             # List of all keywords in the dataset
└── idf.txt                 # IDF (Inverse Document Frequency) values

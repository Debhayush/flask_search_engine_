from flask import Flask, request, jsonify, render_template
import os
import math
from collections import Counter
import re
from num2words import num2words

# --- Config ---
app = Flask(__name__)
DATA_DIR = "data"
NUM_RESULTS = 10

# --- Load data ---
def load_list(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

titles = load_list(os.path.join(DATA_DIR, "problemtitles.txt"))
urls = load_list(os.path.join(DATA_DIR, "problemurls.txt"))
keywords = load_list(os.path.join(DATA_DIR, "keywords.txt"))
idf = [float(x) for x in load_list(os.path.join(DATA_DIR, "IDF.txt"))]
magnitudes = [float(x) for x in load_list(os.path.join(DATA_DIR, "Magnitude.txt"))]

# --- Load dense TFIDF matrix ---
def load_tfidf_matrix(filepath, num_docs, num_keywords):
    matrix = [[0.0] * num_keywords for _ in range(num_docs)]
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            i, j, val = line.strip().split()
            i, j = int(i) - 1, int(j)
            matrix[i][j] = float(val)
    return matrix

tfidf_matrix = load_tfidf_matrix(os.path.join(DATA_DIR, "TFIDF.txt"), len(titles), len(keywords))

# --- Stopwords and cleaner ---
STOPWORDS = set([
    "the", "is", "in", "to", "a", "an", "of", "for", "and", "on", "with", "that", "as", "are", "was",
    "this", "it", "be", "or", "from", "by", "at", "if", "then", "can", "we", "you", "your", "i", "but",
    "have", "has", "had", "not", "do", "does", "did", "so", "such", "these", "those", "he", "she", "they"
])

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    cleaned = []
    for token in tokens:
        if token in STOPWORDS:
            continue
        if token.isdigit():
            cleaned.extend(num2words(token).split())
        else:
            cleaned.append(token)
    return cleaned

# --- Query processing ---
def compute_query_vector(query, keywords, idf):
    tokens = clean_text(query)
    freq = Counter(tokens)
    total = sum(freq.values())

    word_to_index = {word: idx for idx, word in enumerate(keywords)}
    vec = [0.0] * len(keywords)

    matched_keywords = []

    for word, count in freq.items():
        if word in word_to_index:
            idx = word_to_index[word]
            tf = count / total
            vec[idx] = tf * idf[idx]
            matched_keywords.append((word, idx, tf, idf[idx], vec[idx]))

    # Logging matched keywords and their values
    print("\n=== Matched Keywords ===")
    for word, idx, tf, idf_val, tfidf in matched_keywords:
        print(f"Keyword: '{word}' | Index: {idx} | TF: {tf:.4f} | IDF: {idf_val:.4f} | TF-IDF: {tfidf:.4f}")
    
    norm = math.sqrt(sum(x ** 2 for x in vec))
    return vec, norm, tokens

# --- Cosine similarity ---
def cosine_similarity(query_vec, doc_vec, query_norm, doc_norm):
    dot = sum(query_vec[i] * doc_vec[i] for i in range(len(query_vec)))
    if doc_norm == 0 or query_norm == 0:
        return 0.0
    return dot / (query_norm * doc_norm)

# --- Flask routes ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search")
def search():
    query = request.args.get("question", "")
    query_vec, query_norm, query_tokens = compute_query_vector(query, keywords, idf)

    print(f"\n🔍 Searching for: '{query}'\n")

    results = []
    for i in range(len(titles)):
        score = cosine_similarity(query_vec, tfidf_matrix[i], query_norm, magnitudes[i])
        if score > 0:
            print(f"Matched Document #{i + 1} → Title: {titles[i]} | Score: {score:.4f}")
            results.append((i, score))

    if not results:
        print("No matching documents found.\n")

    results.sort(key=lambda x: -x[1])

    print("\n📊 Top Ranked Documents:")
    for rank, (idx, score) in enumerate(results[:NUM_RESULTS], start=1):
        print(f"{rank}. ID: {idx + 1} | Title: {titles[idx]} | Score: {score:.4f}")

    top_results = []
    for idx, score in results[:NUM_RESULTS]:
        try:
            with open(os.path.join(DATA_DIR, "problemstatements", f"problemstatement{idx+1}.txt"), encoding="utf-8") as f:
                statement = f.read().replace("\n", " ")[:300] + "..."
        except:
            statement = ""
        top_results.append({
            "id": idx + 1,
            "title": titles[idx],
            "url": urls[idx],
            "score": round(score, 4),
            "statement": statement
        })

    return jsonify(top_results)

@app.route("/problem/<int:pid>")
def problem(pid):
    title = titles[pid - 1]
    url = urls[pid - 1]
    try:
        with open(os.path.join(DATA_DIR, "problems", f"problemtext{pid}.txt"), encoding="utf-8") as f:
            full_text = f.read()
    except:
        full_text = "(Problem not found)"
    return render_template("problem.html", title=title, url=url, full_text=full_text)

if __name__ == "__main__":
    app.run(debug=True, port=5000)

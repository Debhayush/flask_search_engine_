import os
import math
from collections import Counter
from tfidfgen import clean_text  # assumes clean_text is imported from tfidfgen.py

# --- Load required data ---
def load_keywords(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def load_idf(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [float(line.strip()) for line in f.readlines() if line.strip()]

def load_tfidf_matrix(filepath, num_docs, num_keywords):
    matrix = [[0.0] * num_keywords for _ in range(num_docs)]
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            i, j, val = line.strip().split()
            i, j = int(i) - 1, int(j)
            if i >= num_docs:
                continue
            matrix[i][j] = float(val)
    return matrix

def load_magnitudes(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [float(line.strip()) for line in f.readlines() if line.strip()]

# --- Query processing ---
def compute_query_vector(query, keywords, idf):
    tokens = clean_text(query)
    freq = Counter(tokens)
    total = sum(freq.values())

    word_to_index = {word: idx for idx, word in enumerate(keywords)}
    vec = [0.0] * len(keywords)

    for word, count in freq.items():
        if word in word_to_index:
            idx = word_to_index[word]
            tf = count / total
            vec[idx] = tf * idf[idx]
    
    norm = math.sqrt(sum(x ** 2 for x in vec))
    return vec, norm

# --- Cosine similarity ---
def cosine_similarity(query_vec, doc_vec, query_norm, doc_norm):
    dot = sum(query_vec[i] * doc_vec[i] for i in range(len(query_vec)))
    if doc_norm == 0 or query_norm == 0:
        return 0.0
    return dot / (query_norm * doc_norm)

# --- Helper to explain matches ---
def print_matches_for_query(query_tokens, keywords, tfidf_matrix, idf_vector):
    word_to_index = {word: i for i, word in enumerate(keywords)}
    print("\n🔍 Matching tokens and their TF-IDF weights per doc:")
    for token in query_tokens:
        if token in word_to_index:
            idx = word_to_index[token]
            print(f"\nToken: '{token}' (IDF = {idf_vector[idx]:.4f})")
            for doc_idx, tfidf_vec in enumerate(tfidf_matrix):
                if tfidf_vec[idx] > 0:
                    print(f"  ➤ Doc {doc_idx+1}: TF-IDF = {tfidf_vec[idx]:.4f}")

# --- Main execution ---
if __name__ == "__main__":
    folder_path = "problemdata"
    keywords = load_keywords("keywords.txt")
    idf = load_idf("IDF.txt")
    tfidf_matrix = load_tfidf_matrix("TFIDF.txt", num_docs=5, num_keywords=len(keywords))
    magnitudes = load_magnitudes("Magnitude.txt")[:5]

    query = "pattern array"
    query_vec, query_norm = compute_query_vector(query, keywords, idf)

    print(f"\n📌 Cosine Similarities for Query: '{query}'\n")
    for i in range(5):
        score = cosine_similarity(query_vec, tfidf_matrix[i], query_norm, magnitudes[i])
        print(f"Doc {i+1} → Score: {score:.6f}")

    # 👇 Call after scores are shown
    print_matches_for_query(clean_text(query), keywords, tfidf_matrix, idf)
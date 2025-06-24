import math
import re
from collections import defaultdict, Counter

# Load vocabulary
with open("keywords.txt", encoding="utf-8") as f:
    keywords = [line.strip() for line in f]
word_to_index = {word: idx for idx, word in enumerate(keywords)}

# Load IDF values
with open("IDF.txt") as f:
    idf = [float(line.strip()) for line in f]

# Load problem titles and URLs
with open("problemtitles.txt", encoding="utf-8") as f:
    titles = [line.strip() for line in f]
with open("problemurls.txt", encoding="utf-8") as f:
    urls = [line.strip() for line in f]

# Load Magnitudes
with open("Magnitude.txt") as f:
    magnitudes = [float(line.strip()) for line in f]

# Load sparse TF-IDF matrix
doc_vectors = defaultdict(dict)
with open("TFIDF.txt") as f:
    for line in f:
        doc_idx, word_idx, val = line.strip().split()
        doc_vectors[int(doc_idx)][int(word_idx)] = float(val)

# Function to clean and tokenize query
def preprocess_query(query):
    stop_words = {
        "the", "is", "in", "to", "a", "an", "of", "for", "and", "on", "with", "that", "as", "are", "was",
        "this", "it", "be", "or", "from", "by", "at", "if", "then", "can", "we", "you", "your", "i", "but"
    }
    words = re.findall(r'\b[a-z]+\b', query.lower())
    return [w for w in words if w not in stop_words]

# Create TF-IDF vector for query
def compute_query_vector(query_tokens):
    tf = Counter(query_tokens)
    vec = {}
    for word, freq in tf.items():
        if word in word_to_index:
            idx = word_to_index[word]
            vec[idx] = freq * idf[idx]
    return vec

# Compute cosine similarity
def cosine_similarity(query_vec, doc_vec, doc_mag):
    dot = sum(query_vec.get(j, 0) * doc_vec.get(j, 0) for j in query_vec)
    qmag = math.sqrt(sum(v**2 for v in query_vec.values()))
    if qmag == 0 or doc_mag == 0:
        return 0
    return dot / (qmag * doc_mag)

# MAIN SEARCH FUNCTION
def search(query, top_k=5):
    query_tokens = preprocess_query(query)
    query_vec = compute_query_vector(query_tokens)

    scores = []
    for i in range(1, len(titles)+1):
        sim = cosine_similarity(query_vec, doc_vectors[i], magnitudes[i-1])
        scores.append((sim, i-1))

    top_results = sorted(scores, reverse=True)[:top_k]
    print(f"\n🔍 Top results for: \"{query}\"")
    for score, idx in top_results:
        print(f"{titles[idx]} → {urls[idx]} (Score: {score:.4f})")

# Example:
search("minimum coins")

import os
import re
import string
from num2words import num2words

# Define stopwords
STOPWORDS = {
    "the", "is", "in", "to", "a", "an", "of", "for", "and", "on", "with", "that", "as", "are", "was",
    "this", "it", "be", "or", "from", "by", "at", "if", "then", "can", "we", "you", "your", "i", "but",
    "have", "has", "had", "not", "do", "does", "did", "so", "such", "these", "those", "he", "she", "they"
}

def clean_text(text):
    """Cleans text by lowercasing, removing punctuation, stopwords, and converting digits to words."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = re.findall(r'\b\w+\b', text)

    cleaned = []
    for token in tokens:
        if token in STOPWORDS:
            continue
        if token.isdigit():
            words = num2words(token).split()
            cleaned.extend(words)
        else:
            cleaned.append(token)
    return cleaned

def extract_unique_keywords(folder_path, output_file='keywords.txt'):
    """Processes all .txt files in the folder, returns and saves sorted unique keywords."""
    keyword_set = set()

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tokens = clean_text(content)
                keyword_set.update(tokens)

    sorted_keywords = sorted(keyword_set)

    with open(output_file, 'w', encoding='utf-8') as f:
        for word in sorted_keywords:
            f.write(word + "\n")

    print(f"✅ Saved {len(sorted_keywords)} unique keywords to {output_file}")
    return sorted_keywords

import os
import re
from collections import Counter

def compute_tf_matrix(folder_path, keywords):
    """
    Returns a 2D TF matrix.
    Rows = documents.
    Columns = keywords (ordered as in the input keyword list).
    Each cell contains tf = count(keyword) / total_keywords_in_doc
    """
    word_to_index = {word: idx for idx, word in enumerate(keywords)}
    matrix = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                tokens = clean_text(text)
                total = len(tokens)

                row = [0.0] * len(keywords)
                if total > 0:
                    freqs = Counter(tokens)
                    for word, count in freqs.items():
                        if word in word_to_index:
                            idx = word_to_index[word]
                            row[idx] = count / total

                matrix.append(row)

    print(f"✅ Computed 2D TF matrix for {len(matrix)} documents.")
    return matrix
import math

def compute_idf_vector(tf_matrix, keywords, output_file="IDF.txt"):
    """
    Computes IDF for each keyword using: IDF[i] = 1 + log(N / nt)
    where:
      - N = total number of documents
      - nt = number of documents containing the keyword i
    Also writes the IDF vector to the specified output file.
    
    Returns:
        A list of IDF values (same order as keywords).
    """
    N = len(tf_matrix)
    keyword_count = len(keywords)
    doc_freq = [0] * keyword_count  # nt values

    # Count how many documents each keyword appears in
    for doc in tf_matrix:
        for i, tf_val in enumerate(doc):
            if tf_val > 0:
                doc_freq[i] += 1

    idf = []
    for i in range(keyword_count):
        nt = doc_freq[i]
        val = 1 + math.log10(N / nt) if nt != 0 else 0
        idf.append(val)

    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        for val in idf:
            f.write(f"{val:.6f}\n")

    print(f"✅ IDF vector computed and saved to '{output_file}'")
    return idf
def compute_tfidf_matrix(tf_matrix, idf_vector, output_file="TFIDF.txt"):
    """
    Computes TF-IDF matrix and saves it in sparse format:
    i j tfidf_value — where i is document number, j is keyword index, and tfidf is non-zero value.

    Parameters:
        tf_matrix (list of list of float): Term frequency matrix.
        idf_vector (list of float): IDF values.
        output_file (str): Path to save the sparse TF-IDF matrix.

    Returns:
        tfidf_matrix (list of list of float): 2D TF-IDF matrix.
    """
    num_docs = len(tf_matrix)
    num_keywords = len(idf_vector)

    tfidf_matrix = [[0.0] * num_keywords for _ in range(num_docs)]

    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(num_docs):
            for j in range(num_keywords):
                tfidf_val = tf_matrix[i][j] * idf_vector[j]
                if tfidf_val > 0:
                    tfidf_matrix[i][j] = tfidf_val
                    f.write(f"{i+1} {j} {tfidf_val:.6f}\n")

    print(f"✅ TF-IDF matrix created and saved to '{output_file}'")
    return tfidf_matrix
import math

def compute_magnitude_vector(tfidf_matrix, output_file="Magnitude.txt"):
    """
    Computes magnitude (L2 norm) of TF-IDF vector for each document.
    Saves result to a file and returns the magnitude vector.

    Parameters:
        tfidf_matrix (list of list of float): TF-IDF matrix (dense 2D list).
        output_file (str): Path to output file.

    Returns:
        list of float: Magnitude of each document vector.
    """
    magnitudes = []

    with open(output_file, "w", encoding="utf-8") as f:
        for doc_vector in tfidf_matrix:
            norm = math.sqrt(sum(val ** 2 for val in doc_vector))
            magnitudes.append(norm)
            f.write(f"{norm:.6f}\n")

    print(f"✅ Magnitude vector saved to '{output_file}'")
    return magnitudes
def run_all(folder_path="problemdata", output_dir="."):
    """
    Executes the full TF-IDF pipeline:
    1. Extracts sorted unique keywords
    2. Computes TF matrix
    3. Computes IDF vector
    4. Computes TF-IDF sparse matrix
    5. Computes magnitude vector

    Files created:
        - keywords.txt
        - IDF.txt
        - TFIDF.txt
        - Magnitude.txt
    """
    print("🚀 Running TF-IDF pipeline...\n")

    # Step 1: Extract keywords
    keyword_file = os.path.join(output_dir, "keywords.txt")
    keywords = extract_unique_keywords(folder_path, output_file=keyword_file)

    # Step 2: Compute TF matrix
    tf_matrix = compute_tf_matrix(folder_path, keywords)

    # Step 3: Compute IDF vector
    idf_file = os.path.join(output_dir, "IDF.txt")
    idf_vector = compute_idf_vector(tf_matrix, keywords, output_file=idf_file)

    # Step 4: Compute TF-IDF matrix and write sparse format
    tfidf_file = os.path.join(output_dir, "TFIDF.txt")
    tfidf_matrix = compute_tfidf_matrix(tf_matrix, idf_vector, output_file=tfidf_file)

    # Step 5: Compute magnitudes
    mag_file = os.path.join(output_dir, "Magnitude.txt")
    magnitude_vector = compute_magnitude_vector(tfidf_matrix, output_file=mag_file)

    print("\n🎉 All files generated successfully!")


if __name__ == "__main__":
    run_all(folder_path="problemdata")

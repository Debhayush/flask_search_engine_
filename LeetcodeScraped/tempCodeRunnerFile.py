def load_tfidf_matrix(filepath, num_docs, num_keywords):
#     matrix = [[0.0] * num_keywords for _ in range(num_docs)]
#     with open(filepath, "r", encoding="utf-8") as f:
#         for line in f:
#             i, j, val = line.strip().split()
#             i, j = int(i) - 1, int(j)
#             matrix[i][j] = float(val)
#     return matrix
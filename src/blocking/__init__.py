"""
Blocking / candidate generation module.

Three complementary blockers, unioned (never intersected):
- Lexical: char n-gram MinHash/LSH
- Embedding: multilingual bi-encoder + FAISS ANN
- Address: token overlap on normalized address
"""

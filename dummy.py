import re
from collections import deque
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def tokenize(text, n):
    tokens = []
    words = re.findall(r'\w+', text.lower())  # Extract words using regex
    queue = deque(maxlen=n)  # Use deque to keep track of n-grams
        
    for word in words:
        queue.append(word)
        if len(queue) == n:
            tokens.append(' '.join(queue))
        
    return tokens

def process_file(filename, n):
    with open(filename, 'r') as file:
        text = file.read()
        tokens = tokenize(text, n)
        return tokens

    #--------------------REMOVAL OF STOPWORDS-------------------------
def remove_stopwords(tokens1,tokens2):
    stop_words = set(stopwords.words('english'))

    filtered_ngrams1 = []
    filtered_ngrams2 = []

    for ngram in tokens1:
        tokenized_ngram = word_tokenize(ngram)
        filtered_ngram = [word for word in tokenized_ngram if word.lower() not in stop_words]
        filtered_ngrams1.append(' '.join(filtered_ngram))

    for ngram in tokens2:
        tokenized_ngram = word_tokenize(ngram)
        filtered_ngram = [word for word in tokenized_ngram if word.lower() not in stop_words]
        filtered_ngrams2.append(' '.join(filtered_ngram))
    return filtered_ngrams1, filtered_ngrams2
    #--------------------KEYWORD EXTRACTION-------------------------

def extract_keywords(filtered_ngrams1, filtered_ngrams2):
    tokenized_words = [ filtered_ngrams1, filtered_ngrams2]

    
    documents = [' '.join(tokens) for tokens in tokenized_words]
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    return tfidf_matrix

     #--------------------COSINE SIMILARITY-------------------------
def similarity(tfidf_matrix):
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    plagiarism_score = similarity_score * 100
    plagiarism_score = float(plagiarism_score)  
    return plagiarism_score

    #-------------------------MAIN-----------------------------------
def main(file1, file2):
    f1 = process_file(file1,3)
    f2 = process_file(file2,3)

    fng1, fng2 = remove_stopwords(f1, f2)
    mat = extract_keywords(fng1, fng2)
    sim = similarity(mat)
    return sim

score = main('file1.txt', 'file2.txt')
print(score)
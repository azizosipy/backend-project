# nlp_app/nlp_utils.py
import fitz  # PDF manipulation
from PIL import Image
import pytesseract
import arabic_reshaper
from bidi.algorithm import get_display
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from qalsadi.lemmatizer import Lemmatizer

# Initialize NLP tools
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
ARABIC_STOPWORDS = set(stopwords.words('arabic'))
TRANSLATOR = str.maketrans('', '', string.punctuation + "،")
lemmatizer = Lemmatizer()
stemmer = ISRIStemmer()

OUTPUT_DIR = "extracted_files"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_text_from_page(page):
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    config = "--oem 1 --psm 3 -l ara"
    raw_text = pytesseract.image_to_string(img, config=config)
    reshaped_text = arabic_reshaper.reshape(raw_text)
    return get_display(reshaped_text)

def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        for page_num in range(doc.page_count):
            text = extract_text_from_page(doc.load_page(page_num))
            output_file = os.path.join(OUTPUT_DIR, f"page_{page_num + 1}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)

def preprocess_text(text):
    cleaned_text = text.translate(TRANSLATOR)
    tokens = word_tokenize(cleaned_text)
    filtered_tokens = [word for word in tokens if word not in ARABIC_STOPWORDS and not word.isdigit()]
    return [stemmer.stem(word.lower()) for word in filtered_tokens]

def preprocess_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        return preprocess_text(file.read())

def preprocess_arabic_corpus(directory_path):
    text_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.txt')]
    processed_texts = [preprocess_text_file(f) for f in text_files]
    return processed_texts, text_files

def cosine_similarity_analysis(target_file_path, corpus_directory, sexual_words):
    # Preprocess the target file and the corpus
    target_tokens = preprocess_text_file(target_file_path)
    corpus_tokens, corpus_files = preprocess_arabic_corpus(corpus_directory)
    
    # Combine all text for TF-IDF
    all_texts = [" ".join(target_tokens)] + [" ".join(tokens) for tokens in corpus_tokens]
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Calculate cosine similarities
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    sexual_words_found = []

    # Check for sexual words in each document and collect them
    for idx, similarity in enumerate(cosine_sim):
        if similarity > 0.5:  # Example threshold for similarity
            matched_words = [word for word in corpus_tokens[idx] if word in sexual_words]
            if matched_words:
                sexual_words_found.append({
                    "file": corpus_files[idx],
                    "similarity": similarity,
                    "sexual_words": matched_words
                })
    
    return sexual_words_found

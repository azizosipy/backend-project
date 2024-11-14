import fitz  # For PDF manipulation
from PIL import Image  # For image manipulation
import pytesseract  # For OCR
import arabic_reshaper  # For reshaping Arabic text
from bidi.algorithm import get_display  # For bidirectional text handling
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
import string
from multiprocessing import Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from qalsadi.lemmatizer import Lemmatizer

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
# Directory to store extracted text files
output_dir = "extracted_files"
os.makedirs(output_dir, exist_ok=True)

# Arabic stopwords
arabic_stopwords = set(stopwords.words('arabic'))

# Initialize tools
lemmatizer = Lemmatizer()
stemmer = ISRIStemmer()
translator = str.maketrans('', '', string.punctuation + "،")

### Step 1: Text Extraction from PDF
def process_page(page_num, pdf_path, output_dir):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)

    # Convert page to image
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # OCR configuration for Arabic
    config = "--oem 1 --psm 3 -l ara"
    try:
        raw_text = pytesseract.image_to_string(img, config=config)
    except pytesseract.TesseractNotFoundError:
        print("Tesseract not found. Make sure Tesseract is installed and added to the PATH.")
        return
    except pytesseract.TesseractError as e:
        print(f"Tesseract error: {str(e)}")
        return

    # Reshape Arabic text for correct display
    reshaped_text = arabic_reshaper.reshape(raw_text)
    bidi_text = get_display(reshaped_text)

    # Save extracted text
    output_file = os.path.join(output_dir, f"page_{page_num + 1}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(bidi_text)

    print(f"Saved text from page {page_num + 1} to {output_file}")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    for page_num in range(doc.page_count):
        process_page(page_num, pdf_path, output_dir)

### Step 2: Text Preprocessing
def preprocess_text(text):
    cleaned_text = text.translate(translator)
    tokens = word_tokenize(cleaned_text)
    filtered_tokens = [
        word for word in tokens if word not in arabic_stopwords and not any(char.isdigit() for char in word)
    ]
    root_tokens = [stemmer.stem(word.lower()) for word in filtered_tokens]
    return root_tokens

def preprocess_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return preprocess_text(text)

def preprocess_arabic_corpus(directory_path):
    text_files = [
        os.path.join(directory_path, filename) for filename in os.listdir(directory_path) if filename.endswith('.txt')
    ]
    with Pool() as pool:
        processed_texts = pool.map(preprocess_text_file, text_files)
    return processed_texts, text_files

### Step 3: Cosine Similarity Analysis
def cosine_similarity_analysis(target_file_path, corpus_directory):
    # Preprocess target file
    target_text = preprocess_text_file(target_file_path)

    # Preprocess corpus
    processed_corpus, corpus_files = preprocess_arabic_corpus(corpus_directory)

    # Create TF-IDF matrix
    corpus_texts = [" ".join(tokens) for tokens in processed_corpus]
    all_texts = [" ".join(target_text)] + corpus_texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Compute cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    for idx, similarity in enumerate(cosine_similarities[0]):
        print(f"Cosine similarity between '{target_file_path}' and {corpus_files[idx]}: {similarity:.4f}")

    # Word Occurrence Analysis
    word_occurrences = {}
    for word in target_text:
        word_occurrences[word] = [doc_text.count(word) for doc_text in processed_corpus]

    for word, occurrences in word_occurrences.items():
        if any(count >= 1 for count in occurrences):
            print(f"Occurrences of '{word}':")
            for idx, count in enumerate(occurrences):
                if count >= 1:
                    print(f"  In {corpus_files[idx]}: {count}")

### Execution
if __name__ == "__main__":
    pdf_path = r'C:\Users\oussa\OneDrive\Desktop\annabi\backend-project\pdf_classification\analyser\adult_book.pdf'
    extract_text_from_pdf(pdf_path)

    # Analyze similarity with a specific text file
    file_path = r'C:\Users\oussa\OneDrive\Desktop\annabi\backend-project\pdf_classification\analyser\sexual_words.txt'
    directory_path = r'.\extracted_files'
    cosine_similarity_analysis(file_path, directory_path)

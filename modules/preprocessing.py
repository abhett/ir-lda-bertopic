# modules/preprocessing.py
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import pos_tag, DependencyGraph
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Setup
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Stopwords bawaan + tambahan domain puisi
base_stopwords = set(stopwords.words('indonesian'))

custom_stopwords = {
    'kau', 'aku', 'hati', 'rindu', 'cinta', 'malam', 'pulang', 'mata',
    'sunyi', 'warna', 'puisi', 'senja', 'hitam', 'putih', 'jiwa', 'langit',
    'kan', 'nya', 'lah', 'pun', 'tak', 'engkau', 'kami', 'kita'
}

stop_words = base_stopwords.union(custom_stopwords)

stemmer = StemmerFactory().create_stemmer()

def preprocess(text):
    # Lowercasing
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenization
    tokens = word_tokenize(text)

    # Stopword Removal
    tokens = [t for t in tokens if t not in stop_words and t.isalpha()]

    # Stemming
    stemmed = [stemmer.stem(token) for token in tokens]

    return ' '.join(stemmed)

def pos_tagging(text):
    tokens = word_tokenize(text)
    return pos_tag(tokens)

def dependency_parse(text):
    return "(dependency parsing not implemented in nltk without grammar)"

import json
import nltk
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Auto-download NLTK data
for pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
    nltk.download(pkg, quiet=True)

app = Flask(__name__)
CORS(app)

# NLP setup
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    """Lowercase → tokenize → remove stopwords → stem"""
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens if t.isalpha() and t not in stop_words]
    return " ".join(tokens)

# Load FAQs
with open("faqs.json", "r") as f:
    faqs = json.load(f)

questions    = [faq["question"] for faq in faqs]
answers      = [faq["answer"]   for faq in faqs]
processed_qs = [preprocess(q)   for q in questions]

# Fit TF-IDF once at startup
vectorizer   = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_qs)

@app.route("/")
def home():
    return jsonify({"status": "FAQ Chatbot API running"})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    processed_input = preprocess(user_input)
    input_vec       = vectorizer.transform([processed_input])
    similarities    = cosine_similarity(input_vec, tfidf_matrix).flatten()
    best_idx        = int(np.argmax(similarities))
    best_score      = float(similarities[best_idx])

    if best_score < 0.1:
        return jsonify({
            "answer": "Sorry, I couldn't find a relevant answer. Please rephrase your question.",
            "matched_question": "",
            "confidence": round(best_score, 2)
        })

    return jsonify({
        "answer":           answers[best_idx],
        "matched_question": questions[best_idx],
        "confidence":       round(best_score, 2)
    })

if __name__ == "__main__":
    print("FAQ Chatbot running at http://127.0.0.1:5000")
    app.run(debug=False, port=5000)

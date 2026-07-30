"""
FAQ Chatbot Engine — Task 2
- Load FAQs (topic: AI & Technology)
- Preprocess with NLTK (tokenize, clean, lemmatize)
- Match via TF-IDF + cosine similarity
"""

import json
import re
import string
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FAQ_PATH = Path(__file__).parent / "faqs.json"
SIMILARITY_THRESHOLD = 0.18


def _ensure_nltk_data():
    """Download required NLTK corpora on first run."""
    for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        nltk.download(pkg, quiet=True)


class FAQChatbotEngine:
    def __init__(self, faq_path: Path = FAQ_PATH):
        _ensure_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.faqs = self._load_faqs(faq_path)
        self._build_index()

    def _load_faqs(self, path: Path) -> list[dict]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return [
            {
                "category": item["category"],
                "question": item["q"],
                "answer": item["a"],
            }
            for item in data
        ]

    def preprocess(self, text: str) -> str:
        """
        NLP preprocessing pipeline:
        1. Lowercase & remove URLs/emails
        2. Remove punctuation & digits
        3. Tokenize
        4. Remove stopwords
        5. Lemmatize
        """
        text = text.lower().strip()
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\d+", "", text)

        tokens = word_tokenize(text)
        tokens = [
            self.lemmatizer.lemmatize(t)
            for t in tokens
            if t not in self.stop_words and len(t) > 1 and t.isalpha()
        ]
        return " ".join(tokens)

    def _build_index(self):
        """Build TF-IDF vectors for all FAQ questions + answers."""
        corpus = []
        for faq in self.faqs:
            combined = f"{faq['question']} {faq['answer']}"
            corpus.append(self.preprocess(combined))

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def find_best_match(self, user_question: str) -> dict:
        """Cosine similarity between user query and FAQ corpus."""
        processed_query = self.preprocess(user_question)
        if not processed_query:
            return self._no_match_response()

        query_vec = self.vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        best_idx = int(similarities.argmax())
        best_score = float(similarities[best_idx])

        if best_score < SIMILARITY_THRESHOLD:
            return {
                "answer": (
                    "I couldn't find a close match in our knowledge base. "
                    "Try rephrasing your question or ask about AI, machine learning, "
                    "programming, careers, or technology topics."
                ),
                "matched_question": None,
                "similarity": round(best_score, 4),
                "category": None,
                "confidence": "low",
            }

        faq = self.faqs[best_idx]
        return {
            "answer": faq["answer"],
            "matched_question": faq["question"],
            "similarity": round(best_score, 4),
            "category": faq["category"],
            "confidence": "high" if best_score >= 0.35 else "medium",
        }

    def _no_match_response(self) -> dict:
        return {
            "answer": "Please enter a valid question so I can search our FAQs.",
            "matched_question": None,
            "similarity": 0.0,
            "category": None,
            "confidence": "low",
        }

    def get_answer(self, user_question: str) -> dict:
        return self.find_best_match(user_question)


# Singleton for server
_engine: FAQChatbotEngine | None = None


def get_engine() -> FAQChatbotEngine:
    global _engine
    if _engine is None:
        _engine = FAQChatbotEngine()
    return _engine


if __name__ == "__main__":
    import sys

    engine = FAQChatbotEngine()
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is machine learning?"
    result = engine.get_answer(q)
    print(f"Question: {q}")
    print(f"Match: {result['matched_question']} (score: {result['similarity']})")
    print(f"Answer: {result['answer']}")

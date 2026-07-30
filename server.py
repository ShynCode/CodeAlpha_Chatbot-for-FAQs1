"""
Flask API server for FAQ Chatbot (Task 2)
Run: python server.py
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from faq_engine import get_engine

app = Flask(__name__, static_folder=".")
CORS(app)


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/chatbot.html")
def chatbot_page():
    return send_from_directory(".", "chatbot.html")


@app.route("/faqs.json")
def faqs_data():
    return send_from_directory(".", "faqs.json")


@app.route("/api/health")
def health():
    engine = get_engine()
    return jsonify({
        "status": "ok",
        "faqs_loaded": len(engine.faqs),
        "method": "NLTK preprocessing + TF-IDF cosine similarity",
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "Message is required"}), 400

    engine = get_engine()
    result = engine.get_answer(message)
    return jsonify({
        "message": message,
        "answer": result["answer"],
        "matched_question": result["matched_question"],
        "similarity": result["similarity"],
        "category": result["category"],
        "confidence": result["confidence"],
    })


@app.route("/api/faqs")
def list_faqs():
    engine = get_engine()
    return jsonify({"count": len(engine.faqs), "faqs": engine.faqs[:20]})


if __name__ == "__main__":
    print("Loading FAQ engine (NLTK + cosine similarity)...")
    get_engine()
    print("Server ready → http://127.0.0.1:5000/chatbot.html")
    app.run(host="0.0.0.0", port=5000, debug=False)

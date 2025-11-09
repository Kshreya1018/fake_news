from flask import Flask, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords

with open("fake_model", "rb") as f:
    model_data = pickle.load(f)

tfidf = model_data["tfidf"]
model = model_data["model"]

STOP = set(stopwords.words('english')) - {'no','nor','not'}

def clean_text(x):
    if not isinstance(x, str): 
        x = str(x)
    x = x.lower()
    x = re.sub(r'http\S+|www\S+', ' ', x)
    x = re.sub(r'[^a-z0-9.,!?; ]', ' ', x)
    x = re.sub(r'\s+', ' ', x).strip()
    return " ".join([w for w in x.split() if w not in STOP])

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to the Fake News Detection API with Chatbot",
        "endpoints": {
            "/predict": "POST → { 'text': 'your news text' }",
            "/chatbot": "POST → { 'message': 'your question' }"
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_input = data.get("text", "")

    cleaned = clean_text(user_input)
    vectorized = tfidf.transform([cleaned])
    pred = model.predict(vectorized)[0]
    prob = model.predict_proba(vectorized)[0].max()

    result = "REAL ✅" if pred == 1 else "FAKE ❌"
    confidence = round(prob * 100, 2)

    return jsonify({
        "prediction": result,
        "confidence": f"{confidence}%"
    })

def chatbot_response(message):
    message = message.lower()

    if "hello" in message or "hi" in message:
        return "Hello! 👋 I can check if a news article is real or fake. Just send it to /predict."

    if "how detect" in message or "how to detect" in message:
        return "To detect fake news, look for: unusual website names, emotional headlines, no sources, and grammar mistakes."

    if "what is fake news" in message:
        return "Fake news refers to misinformation spread to mislead people. Always verify sources before believing or sharing."

    if "who made you" in message:
        return "I am built to support a Fake News Detection ML Model. I help classify and explain news credibility."

    return "I can help detect fake news. Send text to analyze it!"

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    user_message = data.get("message", "")
    reply = chatbot_response(user_message)
    return jsonify({"response": reply})


if __name__ == "__main__":
    app.run(debug=True)

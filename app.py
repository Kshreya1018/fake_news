from flask import Flask, request, jsonify
import pickle
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

with open("fake_model", "rb") as f:
    model_data = pickle.load(f)

tfidf = model_data["tfidf"]
model = model_data["model"]

STOP = set(ENGLISH_STOP_WORDS) - {'no','nor','not'}

def clean_text(x):
    if not isinstance(x, str): 
        x = str(x)
    x = x.lower()
    x = re.sub(r'http\S+|www\S+', ' ', x)
    x = re.sub(r'[^a-z0-9.,!?; ]', ' ', x)
    x = re.sub(r'\s+', ' ', x).strip()
    return " ".join([w for w in x.split() if w not in STOP])

def is_news_text(message):
    text = message.lower().strip()

    if len(text.split()) < 6:
        return False

    sentence_count = text.count('.') + text.count('!') + text.count('?')
    if sentence_count > 1:
        return True

    news_keywords = [
        "report", "reports", "said", "according to", "breaking", "headline", 
        "official", "statement", "confirmed", "announced", "investigation",
        "sources", "declared", "revealed"
    ]
    if any(word in text for word in news_keywords):
        return True

    if any(char.isdigit() for char in text):
        return True

    if text.count(',') > 1 or text.count('.') > 1:
        return True

    return False

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to the Fake News Detection API with Chatbot",
        "bot_starter_message": "Hello! 👋 I can help verify news. Send any news text to /predict.",
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
    msg = message.lower()

    if is_news_text(msg):
        return "This looks like a news article 📄. Please send it to /predict so I can check if it's REAL or FAKE."

    if msg in ["hi", "hello", "hey", "start", "hola"]:
        return "Hello! 👋 I can help detect fake news. Send any article or message text to /predict, and I'll analyze it."

    if "true or fake" in msg or "real or fake" in msg or "is this news true" in msg:
        return "Send the news text to /predict, and I will verify it ✅"

    if "verify" in msg:
        return "Sure! Paste the article or message and send it to /predict for verification."

    if "instagram" in msg or "twitter" in msg or "social media" in msg:
        return "News from social media can be misleading 🚫. Always verify using trusted sources or /predict."

    if "whatsapp" in msg or "forward" in msg:
        return "Forwarded messages are often false ❗ Double-check using /predict."

    if "trustworthy" in msg or "reliable sources" in msg:
        return "Reliable news sources include: Reuters, BBC, The Hindu, Associated Press, and official government sites."

    if "why fake news spreads" in msg or "why does fake news spread" in msg:
        return "Fake news spreads quickly because it triggers emotions and gets shared without verification."

    if "signs" in msg or "identify" in msg:
        return "Signs of fake news: sensational headlines, no credible sources, poor grammar, and unknown websites."

    if "images" in msg or "videos" in msg:
        return "Yes, images and videos can be faked. Use Google Lens or InVid to check authenticity 🔍."

    if "clickbait" in msg or "headline" in msg:
        return "Clickbait headlines are made to trigger emotions, not inform. Always read the full article carefully."

    if "accuracy" in msg:
        return "The model is trained on real datasets and performs reliably, but verification from multiple sources is best."

    if "emotional" in msg or "shocking" in msg:
        return "Extreme emotional headlines are often misleading ⚠️. Be cautious."

    if "fact check" in msg or "fact-check" in msg:
        return "Fact-checking means verifying news using credible sources before believing or sharing it."

    if "fact checking websites" in msg or "fact-check sites" in msg:
        return "Try these fact-checkers: Alt News, BOOM Live, FactCheck.org, Reuters Fact Check ✅."

    if "summarize" in msg:
        return "I currently verify fake news. If you'd like, I can help add summarization later 😊."

    if "why believe fake news" in msg:
        return "People believe fake news due to emotional influence, personal bias, and repetition."

    if "how verify news organizations" in msg or "how journalists verify" in msg:
        return "News organizations verify by cross-checking with multiple sources and official statements."

    return "I can help detect fake news 📰. Just send the text to /predict!"

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    user_message = data.get("message", "")
    reply = chatbot_response(user_message)
    return jsonify({"response": reply})

if __name__ == "__main__":
    app.run(debug=True)

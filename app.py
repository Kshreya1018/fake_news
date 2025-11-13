from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

with open("fake_model", "rb") as f:
    model_data = pickle.load(f)

tfidf = model_data["tfidf"]
model = model_data["model"]

STOP = set(ENGLISH_STOP_WORDS) - {'no', 'nor', 'not'}

def clean_text(x):
    if not isinstance(x, str):
        x = str(x)
    x = x.lower()
    x = re.sub(r'http\S+|www\S+', ' ', x)
    x = re.sub(r'[^a-z0-9.,!?; ]', ' ', x)
    x = re.sub(r'\s+', ' ', x).strip()
    return " ".join([w for w in x.split() if w not in STOP])

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to the Fake News Detection API with Chatbot",
        "bot_starter_message": "Hi there! Send me any news text and I’ll help check it 😊",
        "endpoints": {
            "/predict": "POST → { 'message': 'your news text' }",
            "/chatbot": "POST → { 'message': 'your message' }"
        }
    })


@app.route("/predict", methods=["POST"])
def predict_news():
    data = request.get_json()
    news_text = data.get("message", "")
    if not news_text.strip():
        return jsonify({"reply": "Please provide some news text to analyze."})

    cleaned = clean_text(news_text)
    vectorized = tfidf.transform([cleaned])
    pred = model.predict(vectorized)[0]
    prob = model.predict_proba(vectorized)[0].max()

    result = "REAL ✅" if pred == 1 else "FAKE ❌"
    confidence = round(prob * 100, 2)

    return jsonify({
        "reply": f"The news appears to be {result}. Confidence: {confidence}%"
    })


def get_chatbot_reply(message):
    msg = message.lower().strip()

    if msg in ["hi", "hello", "hey", "hii", "hola", "start"]:
        return "Hiii! Send me any news headline or message, and I’ll help you check if it’s real or fake 😊"

    if "thank" in msg:
        return "Aww, you're welcome! 🫶 I'm always here to help 💛"

    if msg in ["who are you", "your name", "what are you"]:
        return "I'm NewsBuddy! Your friendly helper for checking if news is real or fake 😊"

    if "whatsapp" in msg or "forward" in msg:
        return "Those WhatsApp forwards can be tricky 😕 You're doing great by verifying!"

    if "instagram" in msg or "twitter" in msg or "social media" in msg:
        return "Social media spreads news fast — not always accurately 😅 Always good to double-check."

    if "why fake news spreads" in msg or "why does fake news spread" in msg:
        return "Fake news spreads because it triggers emotions and gets shared quickly 😞 But you're being smart by checking!"

    if "reliable sources" in msg or "trusted sources" in msg:
        return "Reliable sources include: Reuters, BBC, The Hindu, Press Information Bureau, and other official sites 🌐"

    if "true or fake" in msg or "real or fake" in msg or "is this news true" in msg:
        return "Sure!! Just paste the news text here, and I’ll analyze it for you."

    if "verify" in msg:
        return "Of course! 😊 Paste the news text, I’ll check it."

    if "signs" in msg or "identify" in msg:
        return "Look for: missing sources, dramatic tone, grammar mistakes, unknown websites 👀 These can be signs of fake news."

    if "images" in msg or "videos" in msg:
        return "Images & videos can be edited 😕 Try Google Lens or InVid to check authenticity 🔍"

    if "clickbait" in msg or "headline" in msg:
        return "Clickbait headlines try to shock you 😯 Always read the full article before believing!"

    if "accuracy" in msg:
        return "The model is trained well ✅ but cross-checking with trusted sources is always wise."

    if "emotional" in msg or "shocking" in msg:
        return "Shocking headlines are often misleading ⚠ You're right to pause and check!"

    if "fact check" in msg or "fact-check" in msg:
        return "Fact-checking means verifying news using credible sources before sharing."

    if "fact checking websites" in msg or "fact-check sites" in msg:
        return "Try: Alt News, BOOM Live, FactCheck.org, Snopes, Reuters Fact Check ✅"

    if "summarize" in msg:
        return "Right now I'm focused on verifying news, but I can learn summarization too if you'd like!"

    if "why believe fake news" in msg:
        return "People believe fake news because it appeals to emotions or biases 😞 That's why verifying is so important."

    if "how verify news organizations" in msg or "how journalists verify" in msg:
        return "Journalists cross-check facts using multiple official & credible sources before publishing 📰"

    return "I’m here to help you verify news. Just send any news text or headline and I’ll help you check if it’s Real or Fake 😊"


@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    user_input = data.get("message", "")
    reply = get_chatbot_reply(user_input)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True)
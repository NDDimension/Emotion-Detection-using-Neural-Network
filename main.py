import streamlit as st
import joblib as jb
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import numpy as np

# Load the model and preprocessing tools
model = tf.keras.models.load_model("emotion_detection.h5")
tokenizer = jb.load("tokenizer.jb")
label_encoder = jb.load("label_encoder.jb")


# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Emotion to emoji mapping
emotion_emojis = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "fear": "😨",
    "surprise": "😲",
    "disgust": "🤢",
    "neutral": "😐",
    "love": "❤️",
    "bored": "🥱",
    "excited": "🤩",
}


# Streamlit UI setup
st.set_page_config(page_title="Emotion Detector", page_icon="😊", layout="centered")

# Main title
st.markdown(
    "<h1 style='text-align: center; color: #FF4B4B;'>💬 Emotion Detector</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; font-size: 18px;'>Discover the emotion behind your words with the power of AI 🤖</p>",
    unsafe_allow_html=True,
)

st.markdown("---")

# Text input
user_input = st.text_area(
    "📝 Enter a sentence below:", height=150, placeholder="Type something here..."
)

# Prediction block
if st.button("🔍 Analyze Emotion"):
    if user_input.strip():
        processed_text = preprocess_text(user_input)
        sequence = tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=100, padding="post")
        prediction = model.predict(padded_sequence)[0]
        predicted_index = np.argmax(prediction)
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]
        confidence = prediction[predicted_index] * 100

        emoji = emotion_emojis.get(predicted_label.lower(), "💡")

        st.markdown(
            f"""
            <div style='
                text-align: center;
                background: linear-gradient(to right, #f9f9f9, #fff);
                padding: 30px;
                border-radius: 15px;
                box-shadow: 2px 4px 20px rgba(0, 0, 0, 0.1);
                margin-top: 30px;
            '>
                <h2 style='color: #4CAF50;'>🎯 Detected Emotion</h2>
                <div style='font-size: 70px;'>{emoji}</div>
                <h3 style='color: #FF4B4B; font-weight: bold;'>{predicted_label.upper()}</h3>
                <p style='font-size: 16px; color: #333;'>Confidence Level</p>
                <div style='
                    background-color: #eee;
                    border-radius: 20px;
                    height: 22px;
                    width: 80%;
                    margin: 0 auto;
                    overflow: hidden;
                '>
                    <div style='
                        background-color: #4CAF50;
                        height: 100%;
                        width: {confidence:.2f}%;
                        text-align: right;
                        border-radius: 20px;
                        line-height: 22px;
                        color: white;
                        padding-right: 10px;
                        font-size: 14px;
                    '>{confidence:.2f}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 14px;'>Made with ❤️ by <strong>Dhanraj Sharma</strong></div>",
    unsafe_allow_html=True,
)

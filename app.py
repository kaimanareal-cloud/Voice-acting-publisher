import streamlit as st
import whisper
import re
from rapidfuzz import fuzz
import tempfile

# -----------------------------
# Load Model (cache so it doesn't reload)
# -----------------------------
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()


# -----------------------------
# Helpers
# -----------------------------
def split_text(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if len(s.strip()) > 0]


def compare(book_sentences, transcription):
    results = []

    for sentence in book_sentences:
        score = fuzz.partial_ratio(sentence.lower(), transcription.lower())

        if score > 90:
            status = "OK"
        elif score > 75:
            status = "LOW MATCH"
        else:
            status = "MISSING"

        results.append((sentence, score, status))

    return results


# -----------------------------
# UI
# -----------------------------
st.title("🎙️ Audiobook QA Checker")
st.write("Upload narration and book text to check for missing or incorrect readings.")

book_text = st.text_area("📄 Paste Book Text")

audio_file = st.file_uploader("🎧 Upload Audio", type=["mp3", "wav"])


# -----------------------------
# Run Button
# -----------------------------
if st.button("🚀 Run Check"):

    if not book_text or not audio_file:
        st.warning("Please provide both book text and audio.")
        st.stop()

    # Save uploaded audio temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_file.read())
        audio_path = tmp.name

    # Transcribe
    with st.spinner("🔍 Transcribing audio..."):
        result = model.transcribe(audio_path)
        transcription = result["text"]

    # Process
    sentences = split_text(book_text)

    with st.spinner("📊 Comparing text..."):
        results = compare(sentences, transcription)

    # -----------------------------
    # Display Results
    # -----------------------------
    st.subheader("📋 Results")

    issues = 0

    for sentence, score, status in results:
        if status == "OK":
            st.markdown(f"✅ {sentence}")
        elif status == "LOW MATCH":
            st.markdown(f"🟡 ({score}%) {sentence}")
            issues += 1
        else:
            st.markdown(f"❌ ({score}%) {sentence}")
            issues += 1

    st.write(f"\n### ⚠️ Total Issues Found: {issues}")
    
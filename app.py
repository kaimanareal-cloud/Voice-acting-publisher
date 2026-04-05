import streamlit as st
import whisper
import re
from rapidfuzz import fuzz
import tempfile
import fitz  # PyMuPDF

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

def extract_text_from_pdf(pdf_file):
    text = ""

    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()

    return text

def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)
    return text

def remove_noise(text):
    lines = text.split("\n")
    clean_lines = [line for line in lines if len(line.strip()) > 30]
    return " ".join(clean_lines)

def compare_with_timestamps(book_sentences, segments):
    results = []

    for sentence in book_sentences:
        best_score = 0
        best_segment = None

        for seg in segments:
            score = fuzz.partial_ratio(sentence.lower(), seg["text"].lower())

            if score > best_score:
                best_score = score
                best_segment = seg

        # Determine status
        if best_score > 90:
            status = "OK"
        elif best_score > 75:
            status = "LOW MATCH"
        else:
            status = "MISSING"

        results.append({
            "sentence": sentence,
            "score": best_score,
            "status": status,
            "start": best_segment["start"] if best_segment else None,
            "end": best_segment["end"] if best_segment else None
        })

    return results

def format_time(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"


# -----------------------------
# UI
# -----------------------------
st.title("🎙️ Audiobook QA Checker")
st.write("Upload narration and book text to check for missing or incorrect readings.")

pdf_file = st.file_uploader("📄 Upload Book PDF", type=["pdf"])

audio_file = st.file_uploader("🎧 Upload Audio", type=["mp3", "wav"])
if audio_file:
    st.audio(audio_file)

# -----------------------------
# Run Button
# -----------------------------
if st.button("🚀 Run Check"):

    if not pdf_file or not audio_file:
        st.warning("Please provide both book PDF and audio.")
        st.stop()

    # Save uploaded audio temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_file.read())
        audio_path = tmp.name

    # Transcribe
    with st.spinner("🔍 Transcribing audio..."):
        result = model.transcribe(audio_path)
        segments = result["segments"]
    
    with st.spinner("📄 Reading PDF..."):
        book_text = extract_text_from_pdf(pdf_file)
    # Process
    
        # 🔧 CLEANUP GOES HERE
    book_text = clean_text(book_text)
    book_text = remove_noise(book_text)
    
    sentences = split_text(book_text)

    with st.spinner("📊 Comparing text..."):
        results = compare_with_timestamps(sentences, segments)

    # -----------------------------
    # Display Results
    # -----------------------------
   

    st.write(book_text[:500])
    
    st.subheader("📋 Results")

    issues = 0

    for r in results:
        if r['start'] is not None:
            timestamp = f"{format_time(r['start'])} - {format_time(r['end'])}"
        else:
            timestamp = "N/A"

        if r["status"] == "OK":
            st.markdown(f"✅ {r['sentence']}")

        elif r["status"] == "LOW MATCH":
            issues += 1
            with st.expander(f"🟡 ({r['score']}%) {timestamp}"):
                st.write(r["sentence"])

        else:
            issues += 1
            with st.expander(f"❌ ({r['score']}%) {timestamp}"):
                st.write(r["sentence"])

st.write(f"### ⚠️ Total Issues Found: {issues}")
    

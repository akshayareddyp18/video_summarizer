import os
import uuid
import streamlit as st
import whisper
import yt_dlp
from moviepy.editor import VideoFileClip
from transformers import pipeline

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Video Summarizer",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------- Load external CSS ----------------
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------- Ensure folders ----------------
os.makedirs("uploads", exist_ok=True)

# ---------------- Load models ----------------
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    asr = whisper.load_model("tiny")
    summarizer = pipeline("summarization", model="Falconsai/text_summarization")
    qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    return asr, summarizer, qa

model, summarizer, qa_model = load_models()

# ---------------- Helpers ----------------
def extract_audio(video_path: str, audio_path: str):
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        raise ValueError("The video has no audio track.")
    clip.audio.write_audiofile(audio_path, logger=None)

def transcribe_audio(audio_path: str) -> str:
    result = model.transcribe(audio_path)
    return result["text"]

def summarize_text(text: str) -> str:
    max_chunk_len = 1000
    if len(text) > max_chunk_len:
        chunks = [text[i:i+max_chunk_len] for i in range(0, len(text), max_chunk_len)]
        summaries = [
            summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]
            for chunk in chunks
        ]
        return " ".join(summaries).strip()
    return summarizer(text, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]

def normalize_youtube_url(url: str) -> str:
    if "youtube.com/shorts/" in url:
        return url.replace("youtube.com/shorts/", "youtube.com/watch?v=")
    if "youtu.be/" in url:
        vid = url.rstrip("/").split("/")[-1]
        return f"https://www.youtube.com/watch?v={vid}"
    return url

def download_youtube_video(url: str) -> str:
    url = normalize_youtube_url(url)
    out_path = os.path.join("uploads", f"video_{uuid.uuid4().hex}.mp4")
    ydl_opts = {"format": "mp4", "outtmpl": out_path, "quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return out_path

# ---------------- Session State ----------------
ss = st.session_state
ss.setdefault("video_path", None)
ss.setdefault("audio_path", None)
ss.setdefault("transcript", "")
ss.setdefault("summary", "")
ss.setdefault("tab", "upload")
ss.setdefault("chat_history", [])

# ---------------- Sidebar ----------------
st.sidebar.title("💬 Chatbot")
enable_chatbot = st.sidebar.checkbox("Enable Chatbot", value=False, key="chatbot_toggle")

# Chat history in sidebar
if enable_chatbot and ss.chat_history:
    st.sidebar.markdown("### 💬 Chat History")
    for msg in ss.chat_history:
        role = "👤" if msg["role"] == "user" else "🤖"
        st.sidebar.write(f"{role} {msg['text']}")

# ---------------- Main Layout ----------------
st.markdown('<h1 class="main-title">Video Summarizer</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="main-subtitle">Transform Hours of Watching into Minutes of Understanding, Effortlessly.</p>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)
with col1:
    if st.button("📁 Upload File", use_container_width=True, key="upload_tab"):
        ss.tab = "upload"
with col2:
    if st.button("🔗 Import Link", use_container_width=True, key="link_tab"):
        ss.tab = "link"

if ss.tab == "upload":
    uploaded = st.file_uploader("", type=["mp4"], label_visibility="collapsed", key="file_uploader")
    if uploaded:
        st.success("✅ File ready to process")
else:
    url = st.text_input("", placeholder="Paste YouTube URL here", label_visibility="collapsed", key="url_input")

st.markdown("<div style='text-align:center; margin-top:20px;'>", unsafe_allow_html=True)
if st.button(f"🔄 {'Process Video' if ss.tab == 'upload' else 'Import and Process'}", key="process_btn"):
    if ss.tab == "upload" and uploaded:
        ss.video_path = os.path.join("uploads", uploaded.name)
        with open(ss.video_path, "wb") as f:
            f.write(uploaded.read())
        ss.audio_path = ss.video_path.replace(".mp4", ".wav")
        try:
            with st.spinner("Extracting audio..."):
                extract_audio(ss.video_path, ss.audio_path)
            st.success("✅ Video processed successfully!")
        except Exception as e:
            st.error(f"Error: {str(e)}")
            ss.video_path, ss.audio_path = None, None
    elif ss.tab == "link" and url.strip():
        try:
            with st.spinner("Downloading video..."):
                ss.video_path = download_youtube_video(url.strip())
            ss.audio_path = ss.video_path.replace(".mp4", ".wav")
            with st.spinner("Extracting audio..."):
                extract_audio(ss.video_path, ss.audio_path)
            st.success("✅ Video processed successfully!")
        except Exception as e:
            st.error(f"Error: {str(e)}")
            ss.video_path, ss.audio_path = None, None
    else:
        st.error("⚠️ Please provide a valid input.")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Transcript + Summary ----------------
if ss.audio_path:
    if st.button("📝 Generate Summary", key="summary_btn"):
        with st.spinner("Transcribing..."):
            ss.transcript = transcribe_audio(ss.audio_path)
        with st.spinner("Summarizing..."):
            ss.summary = summarize_text(ss.transcript)
        st.success("✅ Summary generated!")

    if ss.transcript:
        if st.checkbox("Show Transcript", key="show_transcript"):
            st.markdown('<div class="summary-area">', unsafe_allow_html=True)
            st.subheader("Transcript")
            st.write(ss.transcript)
            st.markdown('</div>', unsafe_allow_html=True)

    if ss.summary:
        st.markdown('<div class="summary-area">', unsafe_allow_html=True)
        st.subheader("Summary")
        st.write(ss.summary)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Chatbot Section ----------------
if enable_chatbot and ss.summary:  # Only show after summary exists
    st.markdown('<div class="chatbot-wrapper">', unsafe_allow_html=True)
    st.markdown('<h3 class="chatbot-title">💬 Chatbot</h3>', unsafe_allow_html=True)

    # Display chat history
    st.markdown('<div class="chatbot-messages">', unsafe_allow_html=True)
    if ss.chat_history:
        for msg in ss.chat_history:
            role_class = "user-msg" if msg["role"] == "user" else "bot-msg"
            role_label = "You" if msg["role"] == "user" else "Bot"
            st.markdown(f"<p class='{role_class}'><b>{role_label}:</b> {msg['text']}</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:gray;'>No messages yet.</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Chat input
    user_input = st.text_input("", key="chat_input", placeholder="Type your message...")
    send_clicked = st.button("Send", key="chat_send")

    if send_clicked and user_input.strip():
        # Add user message
        ss.chat_history.append({"role": "user", "text": user_input})

        # ---------------- QA Model Response ----------------
        # Define default knowledge for general questions
        default_knowledge = """
AI (Artificial Intelligence) is about making machines think and act like humans, 
enabling them to learn, solve problems, and make decisions.
Machine Learning is a subset of AI that allows systems to learn from data.
"""

        # Prepend default knowledge to the video summary
        context_text = default_knowledge + "\n" + ss.summary
        summary_chunks = [context_text[i:i+1000] for i in range(0, len(context_text), 1000)]

        answer = ""
        try:
            for chunk in summary_chunks:
                result = qa_model(question=user_input, context=chunk)
                if result["score"] > 0.1:
                    answer = result["answer"]
                    break
            if not answer:
                answer = "🤖 Sorry, I couldn't find an answer in the video summary."
        except Exception as e:
            answer = f"⚠️ Error generating response: {str(e)}"

        # Add bot response
        ss.chat_history.append({"role": "bot", "text": answer})

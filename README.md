# Smart Video Summarizer

Smart Video Summarizer is a Streamlit-based web application that summarizes long videos into concise text and allows users to chat with the summary. The app works with both uploaded video files and YouTube links. It extracts audio, transcribes speech using Whisper, generates summaries with Transformers, and supports Q&A with a question-answering model.

## Features
- Upload video files
- Import videos from YouTube links
- Automatic speech recognition using Whisper
- Generate text summaries using Transformers
- Ask questions about the video content via chatbot
- Streamlit-based user interface

## Tech Stack
- Frontend: Streamlit
- Speech-to-Text: Whisper
- Summarization: Falconsai/text_summarization (Hugging Face Transformers)
- Question Answering: distilbert-base-cased-distilled-squad
- Video Processing: moviepy, yt-dlp

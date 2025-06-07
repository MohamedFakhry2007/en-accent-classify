import os
import subprocess
import tempfile
import streamlit as st
from moviepy.editor import VideoFileClip
from speechbrain.pretrained import EncoderClassifier

# --- Model and App Configuration ---
MODEL_ID = "Jzuluaga/accent-id-commonaccent_ecapa"
PAGE_TITLE = "Accent Analysis Tool"
PAGE_ICON = "🗣️"

# --- Core Functions ---

@st.cache_resource
def load_model():
    """
    Loads and caches the accent classification model. It will only run the first time.
    """
    return EncoderClassifier.from_hparams(
        source=MODEL_ID,
        savedir=os.path.join(os.getcwd(), "accent-id-model-cache")
    )

def download_video_with_yt_dlp(url, temp_dir):
    """
    Downloads a video and provides detailed error messages if it fails.
    """
    try:
        filepath_template = os.path.join(temp_dir, "video.%(ext)s")
        # We run the command and capture its output
        subprocess.run(
            ["yt-dlp", "-o", filepath_template, url],
            check=True,  # This will raise a CalledProcessError if it fails
            capture_output=True,
            text=True
        )
        downloaded_files = [f for f in os.listdir(temp_dir) if f.startswith("video")]
        if not downloaded_files:
            raise Exception("yt-dlp ran without error but did not produce an output file.")
        return os.path.join(temp_dir, downloaded_files[0])
        
    # --- THIS IS THE CRITICAL IMPROVEMENT ---
    except subprocess.CalledProcessError as e:
        # If yt-dlp fails, we now print its specific error message
        st.error("Video Download Failed. The `yt-dlp` command returned an error:")
        # e.stderr contains the actual error from the command line tool
        st.code(f"yt-dlp error output:\n{e.stderr}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during download: {e}")
        return None

def extract_audio(video_path, audio_path):
    """
    Extracts audio to a 16kHz mono WAV file.
    """
    try:
        with VideoFileClip(video_path) as clip:
            clip.audio.write_audiofile(
                audio_path, fps=16000, nbytes=2, codec='pcm_s16le',
                logger=None, ffmpeg_params=["-ac", "1"]
            )
        return True
    except Exception as e:
        st.error(f"Audio Extraction Failed: {e}")
        return False

def classify_accent(audio_path, classifier):
    """
    Analyzes the audio file and returns the predicted accent and confidence.
    """
    out_prob, score, index, label = classifier.classify_file(audio_path)
    confidence = round(out_prob.max().item() * 100, 2)
    return label[0], confidence

# --- Streamlit User Interface ---

def main():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.markdown("Enter a public video URL (e.g., YouTube, Loom) to analyze the speaker's English accent.")

    classifier = load_model()

    with st.form(key="video_url_form"):
        video_url = st.text_input("Video URL", placeholder="https://www.youtube.com/watch?v=...")
        submit_button = st.form_submit_button(label="Analyze Accent")

    if submit_button and video_url:
        with st.spinner("Processing..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                st.info("Step 1: Downloading video...")
                video_path = download_video_with_yt_dlp(video_url, tmpdir)

                if video_path:
                    st.info("Step 2: Extracting audio...")
                    audio_path = os.path.join(tmpdir, "output_audio.wav")
                    if extract_audio(video_path, audio_path):
                        st.info("Step 3: Analyzing accent...")
                        accent, confidence = classify_accent(audio_path, classifier)
                        
                        st.success("Analysis Complete!")
                        st.subheader("🎯 Results")
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Predicted Accent", accent.capitalize())
                        col2.metric("Confidence", f"{confidence}%")

if __name__ == "__main__":
    main()

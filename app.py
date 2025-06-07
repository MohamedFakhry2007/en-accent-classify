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
ANALYSIS_DURATION_SECONDS = 30 # We will only analyze the first 30 seconds

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
    Downloads a video and provides intelligent error messages if it fails.
    """
    try:
        filepath_template = os.path.join(temp_dir, "video.%(ext)s")
        subprocess.run(
            ["yt-dlp", "-o", filepath_template, url],
            check=True, capture_output=True, text=True, timeout=120
        )
        downloaded_files = [f for f in os.listdir(temp_dir) if f.startswith("video")]
        if not downloaded_files:
            raise Exception("yt-dlp ran without error but did not produce an output file.")
        return os.path.join(temp_dir, downloaded_files[0])
    
    except subprocess.CalledProcessError as e:
        error_message = e.stderr
        st.error("Video Download Failed.")
        if "403" in error_message or "Forbidden" in error_message:
            st.warning(
                "**This is likely due to a block from the video service (e.g., YouTube).** "
                "The server running this app is often blocked by services like YouTube. "
                "Please try a direct `.mp4` link for best results."
            )
        else:
             st.info("The video may be private, deleted, or from an unsupported site.")
        with st.expander("Show Technical Error"):
            st.code(error_message)
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during download: {e}")
        return None

# --- THIS FUNCTION IS NOW MEMORY-EFFICIENT ---
def extract_audio(video_path, audio_path, max_duration_sec):
    """
    Extracts a clip of audio to a 16kHz mono WAV file to save memory.
    """
    try:
        with VideoFileClip(video_path) as clip:
            # Check if the clip is longer than our max duration
            if clip.duration > max_duration_sec:
                # Create a subclip of the first `max_duration_sec` seconds
                audio_clip = clip.subclip(0, max_duration_sec).audio
                st.info(f"For efficiency, only the first {max_duration_sec} seconds of audio will be analyzed.")
            else:
                # Use the full audio if the clip is short
                audio_clip = clip.audio
            
            audio_clip.write_audiofile(
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
    st.markdown("Enter a public video URL to analyze the speaker's English accent.")
    st.info(f"ℹ️ **Note:** Downloads from YouTube may be blocked. Direct MP4 links are most reliable. The first {ANALYSIS_DURATION_SECONDS} seconds of the video will be analyzed.", icon="ℹ️")

    classifier = load_model()

    with st.form(key="video_url_form"):
        video_url = st.text_input("Video URL", placeholder="https://example.com/video.mp4")
        submit_button = st.form_submit_button(label="Analyze Accent")

    if submit_button and video_url:
        with st.spinner("Processing..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                st.info("Step 1: Attempting to download video...")
                video_path = download_video_with_yt_dlp(video_url, tmpdir)

                if video_path:
                    st.success("✅ Download complete!")
                    st.info("Step 2: Extracting audio...")
                    audio_path = os.path.join(tmpdir, "output_audio.wav")
                    # We now pass the max duration to the function
                    if extract_audio(video_path, audio_path, max_duration_sec=ANALYSIS_DURATION_SECONDS):
                        st.info("Step 3: Analyzing accent...")
                        accent, confidence = classify_accent(audio_path, classifier)
                        
                        st.success("Analysis Complete!")
                        st.subheader("🎯 Results")
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Predicted Accent", accent.capitalize())
                        col2.metric("Confidence", f"{confidence}%")

if __name__ == "__main__":
    main()

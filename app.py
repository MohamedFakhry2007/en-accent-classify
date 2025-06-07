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
    Downloads the best stream with audio using the simplest possible command.
    """
    try:
        filepath = os.path.join(temp_dir, "video.mp4")
        command = [
            "yt-dlp",
            "-f", "bestaudio/best",
            "--remux-video", "mp4",
            "-o", filepath,
            url
        ]
        subprocess.run(
            command,
            check=True, capture_output=True, text=True, timeout=120
        )
        if not os.path.exists(filepath):
            raise Exception("yt-dlp ran without error but did not produce an output file.")
        return filepath
    except subprocess.CalledProcessError as e:
        st.error("Video Download Failed.")
        with st.expander("Show Technical Error"):
            st.code(e.stderr)
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during download: {e}")
        return None

# --- THIS FUNCTION CONTAINS THE DEBUGGING CODE ---
def extract_audio(video_path, audio_path, max_duration_sec):
    """
    Extracts audio using a direct call to ffmpeg and shows full debug output on failure.
    """
    st.info("Extracting audio with direct ffmpeg call...")
    try:
        # We still use moviepy for a safe duration check if possible.
        try:
            with VideoFileClip(video_path) as clip:
                duration = clip.duration
        except Exception:
            # If moviepy fails to get duration, we'll proceed without it.
            duration = float('inf') 

        command = [
            "ffmpeg",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
        ]
        
        if duration > max_duration_sec:
            st.info(f"For efficiency, only the first {max_duration_sec} seconds of audio will be analyzed.")
            command.extend(["-t", str(max_duration_sec)])
        
        command.append(audio_path)
        
        # Execute the command and capture its output
        subprocess.run(command, check=True, capture_output=True, text=True)
        
        return True
        
    except subprocess.CalledProcessError as e:
        # --- THIS IS THE CRITICAL DEBUG VISION ---
        st.error("Audio Extraction Failed: The `ffmpeg` command failed.")
        with st.expander("Show ffmpeg Technical Error"):
            st.code(f"FFMPEG STDERR:\n{e.stderr}\n\nFFMPEG STDOUT:\n{e.stdout}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during audio extraction: {e}")
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
    st.info(f"ℹ️ **Note:** The first {ANALYSIS_DURATION_SECONDS} seconds of the video will be analyzed.", icon="ℹ️")

    classifier = load_model()

    with st.form(key="video_url_form"):
        video_url = st.text_input("Video URL", placeholder="https://www.loom.com/share/...")
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
                    if extract_audio(video_path, audio_path, max_duration_sec=ANALYSIS_DURATION_SECONDS):
                        st.success("✅ Audio extracted!")
                        st.info("Step 3: Analyzing accent...")
                        accent, confidence = classify_accent(audio_path, classifier)

                        st.success("Analysis Complete!")
                        st.subheader("🎯 Results")

                        col1, col2 = st.columns(2)
                        col1.metric("Predicted Accent", accent.capitalize())
                        col2.metric("Confidence", f"{confidence}%")

    # --- THIS IS THE NEW FOOTER SECTION ---
    # Add a visual separator line before the footer.
    st.divider()

    # Use markdown with HTML to style and center the footer text.
    st.markdown(
        """
        <div style="text-align: center; color: grey; font-size: 0.85em;">
            <p>Tool by <b>Mohamed Fakhry</b></p>
            <p>For AI Agent Solutions Engineer Candidate Challenge by <b>REMWaste</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

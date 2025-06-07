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

# --- THIS FUNCTION CONTAINS THE DEBUGGING CODE ---
def download_video_with_yt_dlp(url, temp_dir):
    """
    Downloads a video and provides full debug output to diagnose download issues.
    """
    try:
        filepath_template = os.path.join(temp_dir, "video.%(ext)s")
        command = [
            "yt-dlp",
            # We add '--verbose' to see the full decision-making process
            "--verbose",
            "-f", "bestvideo+bestaudio/best",
            "--merge-output-format", "mp4",
            "-o", filepath_template,
            url
        ]
        
        # We run the command and capture its complete output
        result = subprocess.run(
            command,
            check=True, capture_output=True, text=True, timeout=120
        )
        
        # --- THIS IS THE NEW DEBUG VISION ---
        # We display the full log from yt-dlp in an expander for diagnosis.
        with st.expander("Show Full Downloader Log"):
            st.code(result.stdout + "\n" + result.stderr)
            
        downloaded_files = [f for f in os.listdir(temp_dir) if f.startswith("video")]
        if not downloaded_files:
            raise Exception("yt-dlp ran without error but did not produce an output file.")
        return os.path.join(temp_dir, downloaded_files[0])
    
    except subprocess.CalledProcessError as e:
        st.error("Video Download Failed.")
        with st.expander("Show Technical Error"):
             # Show the complete output, which contains the reason for the failure
            st.code(e.stdout + "\n" + e.stderr)
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during download: {e}")
        return None

def extract_audio(video_path, audio_path, max_duration_sec):
    """
    Extracts a clip of audio to a 16kHz mono WAV file.
    """
    try:
        with VideoFileClip(video_path) as clip:
            full_audio = clip.audio
            if full_audio is None:
                raise Exception("The downloaded video file does not contain an audio track.")
            
            if clip.duration > max_duration_sec:
                st.info(f"For efficiency, only the first {max_duration_sec} seconds of audio will be analyzed.")
                audio_to_write = full_audio.subclip(0, max_duration_sec)
            else:
                audio_to_write = full_audio
            
            audio_to_write.write_audiofile(
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
                        st.info("Step 3: Analyzing accent...")
                        accent, confidence = classify_accent(audio_path, classifier)
                        
                        st.success("Analysis Complete!")
                        st.subheader("🎯 Results")
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Predicted Accent", accent.capitalize())
                        col2.metric("Confidence", f"{confidence}%")

if __name__ == "__main__":
    main()

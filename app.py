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

# --- THIS FUNCTION CONTAINS THE SIMPLIFIED DOWNLOAD COMMAND ---
def download_video_with_yt_dlp(url, temp_dir):
    """
    Downloads the best stream with audio using the simplest possible command.
    """
    try:
        # We save directly to a predictable filename to make it easier for moviepy.
        filepath = os.path.join(temp_dir, "video.mp4")
        command = [
            "yt-dlp",
            # THE FIX: This is the most generic format request possible.
            # "bestaudio" will be prioritized, and yt-dlp will handle the container.
            "-f", "bestaudio/best",
            # We explicitly ask for an mp4 output. ffmpeg will convert if needed.
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
        error_message = e.stderr
        st.error("Video Download Failed.")
        if "403" in error_message or "Forbidden" in error_message:
            st.warning(
                "**This is likely due to a block from the video service (e.g., YouTube).** "
                "The server running this app is often blocked. Please try a direct `.mp4` link for best results."
            )
        else:
             st.info("The video may be private, deleted, or from an unsupported site.")
        with st.expander("Show Technical Error"):
            st.code(error_message)
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during download: {e}")
        return None

# --- THIS FUNCTION CONTAINS THE MORE ROBUST EXTRACTION LOGIC ---
def extract_audio(video_path, audio_path, max_duration_sec):
    """
    Extracts a clip of audio to a 16kHz mono WAV file using a more robust method.
    """
    try:
        # Load the entire video file into a clip object first.
        # This is more stable than operating directly on the file path.
        with VideoFileClip(video_path) as clip:
            # Check if the clip's duration is longer than our max
            if clip.duration > max_duration_sec:
                st.info(f"For efficiency, only the first {max_duration_sec} seconds of audio will be analyzed.")
                # Create a subclip from the main clip
                sub_clip = clip.subclip(0, max_duration_sec)
                # Get the audio from the subclip
                audio_to_write = sub_clip.audio
            else:
                # Use the full audio if the clip is short
                audio_to_write = clip.audio

            if audio_to_write is None:
                raise Exception("The video file does not contain an audio track.")

            # Write the final audio object to the file.
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
                    if extract__audio(video_path, audio_path, max_duration_sec=ANALYSIS_DURATION_SECONDS):
                        st.info("Step 3: Analyzing accent...")
                        accent, confidence = classify_accent(audio_path, classifier)

                        st.success("Analysis Complete!")
                        st.subheader("🎯 Results")

                        col1, col2 = st.columns(2)
                        col1.metric("Predicted Accent", accent.capitalize())
                        col2.metric("Confidence", f"{confidence}%")

if __name__ == "__main__":
    main()

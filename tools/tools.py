import os

# Ensure Homebrew binaries (ffmpeg, etc.) are discoverable
if "/opt/homebrew/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")

import numpy
import tempfile
import requests
import whisper
import imageio
import yt_dlp

from PIL import Image
from typing import List, Optional
from urllib.parse import urlparse
from dotenv import load_dotenv
from smolagents import tool, LiteLLMModel
from google import genai
from pytesseract import image_to_string

load_dotenv()

MODEL_ID = "gemini/gemini-2.5-flash"

#  Vision Tool
@tool
def vision_tool(prompt: str, image_list: List[Image.Image]) -> str:
    """Analyzes images with a multimodal model and returns a text answer. Pass local file paths (e.g. from file_from_url) or PIL Image objects in image_list.
    Args:
        prompt (str): The question or task about the images.
        image_list (List[PIL.Image.Image]): A list of PIL Image objects or local file path strings.
    """
    model = LiteLLMModel(model_id=MODEL_ID, api_key=os.getenv("GEMINI_API_KEY"), temperature=0.2)

    # Convert strings (paths or URLs) to PIL Images
    import io
    images = []
    for img in image_list:
        if isinstance(img, str):
            if img.startswith("http://") or img.startswith("https://"):
                headers = {}
                if "huggingface.co" in img:
                    hf_token = os.getenv("HF_TOKEN")
                    if hf_token:
                        headers["Authorization"] = f"Bearer {hf_token}"
                resp = requests.get(img, headers=headers)
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content))
            else:
                img = Image.open(img)
        images.append(img)

    payload = [{"type": "text", "text": prompt}] + [{"type": "image", "image": img} for img in images]
    return model([{"role": "user", "content": payload}]).content


#  YouTube Frame Sampler 
@tool
def youtube_frames_to_images(url: str, every_n_seconds: int = 5) -> List[Image.Image]:
    """Downloads a YouTube video and returns sampled frames as a list of PIL Images. Use with vision_tool to analyze video content visually.
    Args:
        url (str): The YouTube video URL (e.g. https://www.youtube.com/watch?v=...).
        every_n_seconds (int): Seconds between sampled frames (default 5).
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        ydl_cfg = {
            "format": "bestvideo+bestaudio/best",
            "outtmpl": os.path.join(temp_dir, "yt_video.%(ext)s"),
            "merge_output_format": "mp4",
            "quiet": True,
            "force_ipv4": True
        }
        with yt_dlp.YoutubeDL(ydl_cfg) as ydl:
            ydl.extract_info(url, download=True)

        video_file = next((os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.mp4')), None)
        reader = imageio.get_reader(video_file)
        fps = reader.get_meta_data().get("fps", 30)
        interval = int(fps * every_n_seconds)

        return [Image.fromarray(frame) for i, frame in enumerate(reader) if i % interval == 0]


#  YouTube QA via File URI 
@tool
def ask_youtube_video(url: str, question: str) -> str:
    """Answers a question about a YouTube video by sending it to a multimodal model. Returns a concise text answer.
    Args:
        url (str): The YouTube video URL (e.g. https://www.youtube.com/watch?v=...).
        question (str): The question to ask about the video content.
    """

    try:
        # Strip litellm "gemini/" prefix for direct API use
        raw_model = MODEL_ID.replace("gemini/", "")
        client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
        # Prepend context so Gemini always analyzes the actual video
        full_question = (
            f"Based on the content of this video, answer the following question concisely. "
            f"Give ONLY the direct answer with no preamble or explanation: {question}"
        )
        response = client.models.generate_content(
            model=raw_model,
            contents=[full_question, genai.types.Part.from_uri(file_uri=url, mime_type="video/*")],
        )
        return response.text
    except Exception as e:
        return f"Error asking {MODEL_ID} about video: {str(e)}"


#  File Reading Tool 
@tool
def read_text_file(file_path: str) -> str:
    """Reads and returns the full text content of a local file. Use the absolute path (e.g. from file_from_url output).
    Args:
        file_path (str): The absolute path to the text file (e.g. /tmp/myfile.txt).
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


#  File Downloader 
@tool
def file_from_url(url: str, save_as: Optional[str] = None) -> str:
    """Downloads a file from a URL and returns the absolute local file path where it was saved (e.g. '/tmp/file.png'). Always capture the return value and pass it to other tools.
    Args:
        url (str): The URL of the file to download.
        save_as (Optional[str]): Optional filename to save as (just the name, not a path).
    """
    try:
        if not save_as:
            parsed = urlparse(url)
            save_as = os.path.basename(parsed.path) or f"file_{os.urandom(4).hex()}"

        file_path = os.path.join(tempfile.gettempdir(), save_as)
        headers = {}
        if "huggingface.co" in url:
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)

        return file_path
    except Exception as e:
        return f"Download failed: {e}"


#  Audio Transcription (YouTube) 
@tool
def transcribe_youtube(yt_url: str) -> str:
    """Transcribes the spoken audio from a YouTube video into text using Whisper. Returns the full transcript as a string.
    Args:
        yt_url (str): The YouTube video URL (e.g. https://www.youtube.com/watch?v=...).
    """
    model = whisper.load_model("small")

    with tempfile.TemporaryDirectory() as tempdir:
        ydl_opts = {
            "format": "bestaudio",
            "outtmpl": os.path.join(tempdir, "audio.%(ext)s"),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav"
            }],
            "quiet": True,
            "force_ipv4": True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(yt_url, download=True)

        wav_file = next((os.path.join(tempdir, f) for f in os.listdir(tempdir) if f.endswith(".wav")), None)
        return model.transcribe(wav_file)['text']


#  Audio File Transcriber 
@tool
def audio_to_text(audio_path: str) -> str:
    """Transcribes a local audio file (mp3, wav, etc.) into text using Whisper. Use file_from_url first to download the audio file, then pass the returned path here.
    Args:
        audio_path (str): The absolute local path to the audio file (e.g. /tmp/audio.mp3).
    """
    try:
        model = whisper.load_model("small")
        result = model.transcribe(audio_path)
        return result['text']
    except Exception as e:
        return f"Failed to transcribe: {e}"


#  OCR 
@tool
def extract_text_via_ocr(image_path: str) -> str:
    """Extracts text from an image using OCR (Tesseract). Returns the recognized text. Use file_from_url first to download the image, then pass the returned path here.
    Args:
        image_path (str): The absolute local path to the image file (e.g. /tmp/image.png).
    """
    try:
        img = Image.open(image_path)
        return image_to_string(img)
    except Exception as e:
        return f"OCR failed: {e}"


#  CSV Analyzer 
@tool
def summarize_csv_data(path: str, query: str = "") -> str:
    """Loads a CSV file and returns its row count, column names, and summary statistics. Use file_from_url first to download the CSV, then pass the returned path here.
    Args:
        path (str): The absolute local path to the CSV file (e.g. /tmp/data.csv).
        query (str): Optional query or description of what to look for.
    """
    try:
        import pandas as pd
        df = pd.read_csv(path)
        return f"Loaded CSV with {len(df)} rows. Columns: {list(df.columns)}\n\n{df.describe()}"
    except Exception as e:
        return f"CSV error: {e}"


#  Excel Analyzer 
@tool
def summarize_excel_data(path: str, query: str = "") -> str:
    """Loads an Excel file (.xls/.xlsx) and returns its row count, column names, and summary statistics. Use file_from_url first to download the file, then pass the returned path here.
    Args:
        path (str): The absolute local path to the Excel file (e.g. /tmp/data.xlsx).
        query (str): Optional query or description of what to look for.
    """
    try:
        import pandas as pd
        df = pd.read_excel(path)
        return f"Excel file with {len(df)} rows. Columns: {list(df.columns)}\n\n{df.describe()}"
    except Exception as e:
        return f"Excel error: {e}"

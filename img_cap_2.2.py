import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration, BartTokenizer, BartForConditionalGeneration
import easyocr
from gtts import gTTS
import os
import torch
from PIL import Image
import numpy as np
import platform

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Initialize the models and reader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# Initialize the BART summarization model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
summarizer_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)


def generate_caption(image):
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def extract_text_from_image(image):
    if isinstance(image, Image.Image):
        image = np.array(image)

    if not isinstance(image, np.ndarray):
        raise ValueError("Image input must be a numpy array or PIL image")

    result = reader.readtext(image)
    text = ' '.join([res[1] for res in result]) if result else "No text detected"
    return text


def summarize_text(text):
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True).to(device)
    summary_ids = summarizer_model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0,
                                            num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def convert_to_speech(text):
    tts = gTTS(text)
    tts.save("output.mp3")

    # Cross-platform audio playback
    if platform.system() == "Windows":
        os.system("start output.mp3")
    elif platform.system() == "Darwin":  # macOS
        os.system("afplay output.mp3")
    else:  # Linux
        os.system("mpg321 output.mp3")


def process_frame(frame, caption_list, text_list):
    caption = generate_caption(frame)
    extracted_text = extract_text_from_image(frame)
    caption_list.append(caption)
    text_list.append(extracted_text)


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    caption_list, text_list = [], []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 60 == 0:
            process_frame(frame, caption_list, text_list)

        frame_count += 1

    cap.release()
    return caption_list, text_list


def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    caption_list, text_list = [], []
    process_frame(image, caption_list, text_list)
    return caption_list, text_list


def main(input_path):
    if input_path.endswith(('.mp4', '.avi', '.mov')):
        caption_list, text_list = process_video(input_path)
    else:
        caption_list, text_list = process_image(input_path)

    caption_paragraph = ' '.join(caption_list)
    text_paragraph = ' '.join(text_list)

    print("\nFull Captions Paragraph:\n", caption_paragraph)
    print("\nFull Text Paragraph:\n", text_paragraph)

    summarized_captions = summarize_text(caption_paragraph)
    summarized_text = summarize_text(text_paragraph)

    print("\nSummarized Captions:\n", summarized_captions)
    print("\nSummarized Text:\n", summarized_text)

    final_summary = f"Summary of Captions: {summarized_captions}. Summary of Text: {summarized_text}"
    print("\nFinal Combined Summary:\n", final_summary)

    convert_to_speech(final_summary)


# Run with an example file path
main("wallpaper1.jpg")  # Replace with an appropriate path

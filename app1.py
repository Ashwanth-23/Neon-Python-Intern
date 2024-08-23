import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
from transformers import pipeline

st.title("Image Description and Emotion Analysis")

st.write("Enter the URL of an image to get its description and emotiomnal tone.")
def fetch_image(image_url):
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        st.error(f"Error fetching image: {e}")
        return None
def generate_description(image):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    return description
def analyze_emotion(text):
    # Use device=-1 for CPU, device=0 for GPU
    emotion_classifier = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=-1)
    result = emotion_classifier(text)
    return result[0]['label'], result[0]['score']

def process_image(image_url):
    image = fetch_image(image_url)
    if image:
        description = generate_description(image)
        emotion, score = analyze_emotion(description)
        return description, emotion, score
    return None, None, None

# User input for image URL
image_url = st.text_input("Image URL")

if st.button("Process Image"):
    if image_url:
        with st.spinner("Processing..."):
            description, emotion, score = process_image(image_url)
            if description:
                st.image(fetch_image(image_url), caption="Uploaded Image", use_column_width=True)
                st.write(f"**Description:** {description}")
                st.write(f"**Emotion:** {emotion} (Confidence: {score:.2f})")
            else:
                st.error("Failed to process the image. Please check the URL and try again.")
    else:
        st.error("Please provide a valid image URL.")


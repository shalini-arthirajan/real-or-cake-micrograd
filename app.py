from micrograd.nn import MLP
import json
from PIL import Image
import numpy as np
import gradio as gr
import math

#loading model

IMAGE_SIZE = 8

model = MLP(192, [16, 8, 1])

with open("weights.json", "r") as f:
    weights = json.load(f)

for p, w in zip(model.parameters(), weights):
    p.data = float(w)


def preprocess(img):
    img = img.convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

    pixels = np.array(img, dtype=np.float32).flatten() / 255.0
    return pixels.tolist()


def predict(image):
    x = preprocess(image)

    score = model(x).data

    # output onfidence
    confidence = 1 / (1 + math.exp(-score))

    return {
        "Cake": confidence,
        "Real": 1 - confidence,
    }



theme = gr.themes.Soft(
    primary_hue="pink",
    secondary_hue="orange",
    neutral_hue="stone",
)


# app

demo = gr.Interface(
    fn = predict,

    inputs=gr.Image(
        type="pil",
        label="Upload an image "
    ),

    outputs=gr.Label(
        num_top_classes=2,
        label="Prediction"
    ),

    title="Real or Cake?",

    description="""
    Can a terrible neural network built from scratch tell the difference between a real object and a cake? (based on the viral "Real or Cake" trend)

 A binary image classifier built on top of Andrej Karpathy's Micrograd and trained on a tiny handmade dataset of 100 images.
 
 Upload an image and see what the model thinks! (it's probably very wrong)
""",

    theme=theme
)

demo.launch()
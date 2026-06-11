# Real or Cake?

Test it here : https://huggingface.co/spaces/build-small-hackathon/Worst-Cake-Classifier

A terrible binary image classifier built on top of Andrej Karpathy's Micrograd and trained on a tiny handmade dataset of 100 images.

The output is probably confidently wrong

## Why is it terrible?
- dataset was frankly too small (100 images - 50 real objects (from shoes to plants) and 50 yummy cakes)
- all images were resized to 8x8 px, leaving out a lot of visual information for the model
- relies solely on micrograd's multi-layer perceptron (mlp) instead of convolutional neural networks or deep learning libarires like pytorch

## Accuracy

Training: 70%
Testing: 60%

## Run

pip install -r requirements.txt
python app.py

weights.json is already included in the repository, so you don't need to retrain the model before launching the app
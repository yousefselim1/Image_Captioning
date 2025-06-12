Image Captioning Project README
This project implements an image captioning system using multiple deep learning models to generate descriptive captions for images from the Flickr8k dataset. It includes implementations using LSTM, GRU, BART, GPT-2, and a Flask-based web interface for real-time caption generation with ViT-GPT2 and BLIP models. Below is an overview of the project, its components, setup instructions, and usage details.

Table of Contents

Project Overview
Dataset
Models
LSTM
GRU
BART
GPT-2
Flask Web Interface


Installation
Usage
Evaluation
File Structure
Dependencies
Running the Flask App
Limitations
Future Improvements
License


Project Overview
The project aims to generate descriptive captions for images using deep learning models. It leverages pre-trained convolutional neural networks (CNNs) like ResNet-152 for feature extraction and recurrent neural networks (RNNs) like LSTM and GRU, as well as transformer-based models like BART and GPT-2 for caption generation. A Flask web interface is provided for interactive caption generation using ViT-GPT2 and BLIP models, with BLEU score evaluation to compare generated captions.

Dataset
The project uses the Flickr8k dataset, which consists of:

Images: 8,091 images.
Captions: Each image has 5 human-annotated captions stored in captions.txt.
Source: The dataset is downloaded using the kagglehub library from the dataset adityajn105/flickr8k.
Preprocessing:
Captions are converted to lowercase, stripped of non-alphabetic characters, and tokenized with startseq and endseq tokens.
Images are resized to 600x600 (or 224x224 for transformer models) and normalized using ResNet’s mean and standard deviation.




Models
LSTM

Architecture:
Uses ResNet-152 to extract 2048-dimensional image features.
Features are passed through a fully connected layer to reduce dimensionality to 256.
Captions are tokenized and embedded into a 256-dimensional space.
An LSTM layer processes the concatenated image features and caption embeddings.
Two fully connected layers with dropout predict the next word in the sequence.


Training:
Trained for 50 epochs with a batch size of 64.
Optimizer: Adam (learning rate = 1e-3).
Loss: CrossEntropyLoss.


Evaluation:
Generates captions for test images and computes BLEU-1, BLEU-2, and BLEU-4 scores.
Example: For image 1019604187_d087bf9a5f.jpg, captions are generated and visualized with BLEU scores.



GRU

Architecture:
Similar to the LSTM model, but replaces the LSTM layer with a GRU layer for faster training and comparable performance.
Uses the same ResNet-152 feature extraction pipeline and preprocessing steps.


Training:
Same as LSTM: 50 epochs, batch size of 64, Adam optimizer, and CrossEntropyLoss.


Evaluation:
Generates captions for test images (e.g., 1026685415_0431cbf574.jpg) and computes BLEU scores.



BART

Architecture:
Uses a pre-trained VisionEncoderDecoderModel (nlpconnect/vit-gpt2-image-captioning) with ViT as the encoder and BART-like decoder (though the code uses GPT-2 as the decoder in practice).
ResNet-152 extracts image features, which are saved for efficiency.


Processing:
Images are processed with ViTImageProcessor for compatibility with the model.
Captions are generated using beam search (num_beams=4, max_length=16).


Evaluation:
Generates captions for a subset of 100 images.
Visualizes original and predicted captions for 30 images.
Computes loss using CrossEntropyLoss and BLEU scores for evaluation.



GPT-2

Architecture:
Uses the same VisionEncoderDecoderModel (nlpconnect/vit-gpt2-image-captioning) as BART, combining ViT for image encoding and GPT-2 for caption generation.
ResNet-152 extracts image features for consistency with other models.


Processing:
Images are preprocessed and captions generated similarly to BART.


Evaluation:
Generates captions for all images in the dataset.
Visualizes 10 random examples with their generated captions.
Saves model, processor, and tokenizer for reuse.



Flask Web Interface

Purpose: Provides an interactive web interface for generating captions using ViT-GPT2 and BLIP models.
Features:
Upload an image via a web form.
Generate captions using ViT-GPT2 and BLIP models.
Compute BLEU scores to compare captions from both models.
Display the image with generated captions and BLEU scores.


Endpoints:
/predict_vit_gpt2: Generates captions using ViT-GPT2.
/predict_blip: Generates captions using BLIP.
/calculate_bleu: Computes BLEU scores between ViT-GPT2 and BLIP captions.
/: Renders the HTML interface for image upload and caption display.


Frontend:
Built with HTML, CSS, and JavaScript.
Includes image preview, loading indicator, and styled caption display.


Backend:
Uses Flask to handle API requests and serve the web interface.
Supports base64-encoded image uploads for compatibility.




Installation
To set up the project, follow these steps:

Clone the Repository (if applicable):
git clone <repository_url>
cd image-captioning-project


Install Dependencies: Ensure you have Python 3.8+ installed. Install the required packages:
pip install -r requirements.txt


Download NLTK Data:
import nltk
nltk.download('punkt')


Download the Flickr8k Dataset: The dataset is automatically downloaded using kagglehub:
import kagglehub
path = kagglehub.dataset_download("adityajn105/flickr8k")


Set Up Environment:

Ensure CUDA is available for GPU acceleration (optional but recommended).
Verify the dataset path: /kaggle/input/flickr8k for images and captions.





Evaluation

Metrics:
BLEU Scores: Used to evaluate caption quality by comparing generated captions to ground-truth captions.
BLEU-1, BLEU-2, and BLEU-4 are computed for LSTM and GRU models.
Average BLEU scores are calculated for BART and GPT-2.
BLEU scores are also computed to compare ViT-GPT2 and BLIP captions in the Flask app.


Loss: CrossEntropyLoss is used during training (LSTM, GRU) and evaluation (BART).


Visualization:
LSTM/GRU: Displays the test image with its generated caption.
BART/GPT-2: Visualizes multiple images with original and predicted captions.
Flask: Shows the uploaded image with captions from both models and their BLEU scores.




File Structure
image-captioning-project/
├── app.py                   # Flask application script
├── image_features_flickr8k.pkl  # Saved ResNet-152 features
├── generated_captions_flickr8k.pkl  # Saved captions (BART/GPT-2)
├── generated_captions_sample.pkl  # Sample captions (BART)
├── requirements.txt         # List of dependencies
├── README.md                # This file
└── /kaggle/input/flickr8k/  # Dataset directory (downloaded via kagglehub)
    ├── Images/              # Image files
    └── captions.txt         # Caption annotations


Dependencies
The project requires the following Python packages:
torch
torchvision
transformers
pandas
numpy
tqdm
matplotlib
seaborn
tensorflow
keras
nltk
kagglehub
flask
pycocotools
pillow
scikit-learn









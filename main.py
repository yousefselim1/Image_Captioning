import os
import re
import random
import pickle
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import tensorflow as tf
from textwrap import wrap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.utils import Sequence, plot_model, to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50, ResNet152V2, VGG16, DenseNet201, EfficientNetB7
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, GRU, add, Concatenate, Reshape, concatenate, Bidirectional
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, TimeDistributed, Activation, Dropout, Flatten, Dense, Input, Layer
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu
import kagglehub


# Download the data
path = kagglehub.dataset_download("adityajn105/flickr8k")

print("Path to dataset files:", path)

image_path = '/kaggle/input/flickr8k/Images' # path of the images
df = pd.read_csv('/kaggle/input/flickr8k/captions.txt') # path of the dataset
df.head()


warnings.filterwarnings('ignore')

df = pd.read_csv('/kaggle/input/flickr8k/captions.txt')

def text_preprocessing(data):
    data['caption'] = data['caption'].apply(lambda x: x.lower())
    data['caption'] = data['caption'].apply(lambda x: re.sub(r'[^a-z\s]', '', x))
    data['caption'] = data['caption'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    data['caption'] = data['caption'].apply(lambda x: ' '.join([word for word in x.split() if len(word) > 1 or word in ['a', 'i']]))
    data['caption'] = data['caption'].apply(lambda x: 'startseq ' + x + ' endseq')
    return data

df = text_preprocessing(df)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['caption'].tolist())
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(c.split()) for c in df['caption'])

train_images, val_images = train_test_split(df['image'].unique(), test_size=0.15, random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet = models.resnet152(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # remove the final FC layer
resnet.to(device)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

features = {}

image_path = '/kaggle/input/flickr8k/Images'
for image in tqdm(df['image'].unique()):
    img = Image.open(os.path.join(image_path, image)).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = resnet(img_tensor).squeeze().cpu().numpy()
    features[image] = feature


class CaptionDataset(Dataset):
    def __init__(self, df, tokenizer, features, max_length, vocab_size):
        self.data = df
        self.tokenizer = tokenizer
        self.features = features
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.samples = self._create_samples()

    def _create_samples(self):
        samples = []
        for _, row in self.data.iterrows():
            image = row['image']
            caption = row['caption']
            seq = self.tokenizer.texts_to_sequences([caption])[0]
            for i in range(1, len(seq)):
                in_seq = pad_sequences([seq[:i]], maxlen=self.max_length)[0]
                out_seq = seq[i]
                samples.append((self.features[image], in_seq, out_seq))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X_img, X_seq, y = self.samples[idx]
        return (
            torch.tensor(X_img, dtype=torch.float32),
            torch.tensor(X_seq, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )


class CaptionModel(nn.Module):
    def __init__(self, vocab_size, max_length):
        super(CaptionModel, self).__init__()
        self.img_fc = nn.Linear(2048, 256)
        self.embedding = nn.Embedding(vocab_size, 256)
        self.lstm = nn.LSTM(256, 256, batch_first=True)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, image, sequence):
        img_feat = torch.relu(self.img_fc(image)).unsqueeze(1)
        seq_embed = self.embedding(sequence)
        merged = torch.cat((img_feat, seq_embed), dim=1)
        lstm_out, _ = self.lstm(merged)
        x = self.dropout(lstm_out[:, -1, :])
        x = x + img_feat.squeeze(1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        return out

train_dataset = CaptionDataset(df[df['image'].isin(train_images)], tokenizer, features, max_length, vocab_size)
val_dataset = CaptionDataset(df[df['image'].isin(val_images)], tokenizer, features, max_length, vocab_size)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

model = CaptionModel(vocab_size, max_length).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


for epoch in range(50):
    model.train()
    running_loss = 0.0
    for X_img, X_seq, y in tqdm(train_loader):
        X_img, X_seq, y = X_img.to(device), X_seq.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X_img, X_seq)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {running_loss / len(train_loader):.4f}")


def generate_caption(model, tokenizer, feature, max_length):
    model.eval()
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        seq_tensor = torch.tensor(sequence, dtype=torch.long).to(device)
        feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(feature_tensor, seq_tensor)
            yhat = output.argmax(1).item()

        word = next((w for w, idx in tokenizer.word_index.items() if idx == yhat), None)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return ' '.join(in_text.split()[1:])


# Handle the image before enter to the generate_caption function and print the caption and plot the image
def predict_one_image(image_path, model, tokenizer, max_length):
    # Load image
    img = Image.open(image_path).convert('RGB')

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Extract features from ResNet
    with torch.no_grad():
        feature = resnet(img_tensor).squeeze().cpu().numpy()

    # Generate caption
    caption = generate_caption(model, tokenizer, feature, max_length)

    # Display results
    print("Predicted Caption:", caption)
    plt.imshow(img)
    plt.axis('off')
    plt.title(caption)
    plt.show()

    return caption


def calculate_bleu_score(image_name, model, tokenizer, max_length, df, image_dir):
    # Get reference captions for the image
    references = df[df['image'] == image_name]['caption'].tolist()
    references = [ref.split() for ref in references]  # tokenize

    # Get predicted caption
    predicted = predict_one_image(os.path.join(image_dir, image_name), model, tokenizer, max_length)
    candidate = predicted.split()  # tokenize

    # BLEU calculation
    smoothie = SmoothingFunction().method4
    bleu1 = sentence_bleu(references, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu2 = sentence_bleu(references, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu4 = sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    print(f"BLEU-1: {bleu1:.4f}, BLEU-2: {bleu2:.4f}, BLEU-4: {bleu4:.4f}")
    return bleu1, bleu2, bleu4


image_name = "1019604187_d087bf9a5f.jpg"
image_dir = "/kaggle/input/flickr8k/Images"

bleu1, bleu2, bleu4 = calculate_bleu_score(image_name=image_name, model=model, tokenizer=tokenizer, max_length=max_length, df=df, image_dir=image_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet = models.resnet152(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # remove the final FC layer
resnet.to(device)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

df = pd.read_csv('/kaggle/input/flickr8k/captions.txt')

def text_preprocessing(data):
    data['caption'] = data['caption'].apply(lambda x: x.lower())
    data['caption'] = data['caption'].apply(lambda x: re.sub(r'[^a-z\s]', '', x))
    data['caption'] = data['caption'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    data['caption'] = data['caption'].apply(lambda x: ' '.join([word for word in x.split() if len(word) > 1 or word in ['a', 'i']]))
    data['caption'] = data['caption'].apply(lambda x: 'startseq ' + x + ' endseq')
    return data

df = text_preprocessing(df)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['caption'].tolist())
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(c.split()) for c in df['caption'])

train_images, val_images = train_test_split(df['image'].unique(), test_size=0.15, random_state=42)

features = {}

image_path = '/kaggle/input/flickr8k/Images'
for image in tqdm(df['image'].unique()):
    img = Image.open(os.path.join(image_path, image)).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = resnet(img_tensor).squeeze().cpu().numpy()
    features[image] = feature

class CaptionDataset(Dataset):
    def __init__(self, df, tokenizer, features, max_length, vocab_size):
        self.data = df
        self.tokenizer = tokenizer
        self.features = features
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.samples = self._create_samples()

    def _create_samples(self):
        samples = []
        for _, row in self.data.iterrows():
            image = row['image']
            caption = row['caption']
            seq = self.tokenizer.texts_to_sequences([caption])[0]
            for i in range(1, len(seq)):
                in_seq = pad_sequences([seq[:i]], maxlen=self.max_length)[0]
                out_seq = seq[i]
                samples.append((self.features[image], in_seq, out_seq))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X_img, X_seq, y = self.samples[idx]
        return (
            torch.tensor(X_img, dtype=torch.float32),
            torch.tensor(X_seq, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )

class CaptionModel(nn.Module):
    def __init__(self, vocab_size, max_length):
        super(CaptionModel, self).__init__()
        self.img_fc = nn.Linear(2048, 256)
        self.embedding = nn.Embedding(vocab_size, 256)
        self.gru = nn.GRU(256, 256, batch_first=True)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, image, sequence):
        img_feat = torch.relu(self.img_fc(image)).unsqueeze(1)
        seq_embed = self.embedding(sequence)
        merged = torch.cat((img_feat, seq_embed), dim=1)
        gru_out, _ = self.gru(merged)
        x = self.dropout(gru_out[:, -1, :])
        x = x + img_feat.squeeze(1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        return out


train_dataset = CaptionDataset(df[df['image'].isin(train_images)], tokenizer, features, max_length, vocab_size)
val_dataset = CaptionDataset(df[df['image'].isin(val_images)], tokenizer, features, max_length, vocab_size)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

model = CaptionModel(vocab_size, max_length).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(50):
    model.train()
    running_loss = 0.0
    for X_img, X_seq, y in tqdm(train_loader):
        X_img, X_seq, y = X_img.to(device), X_seq.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X_img, X_seq)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {running_loss / len(train_loader):.4f}")

def generate_caption(model, tokenizer, feature, max_length):
    model.eval()
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        seq_tensor = torch.tensor(sequence, dtype=torch.long).to(device)
        feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(feature_tensor, seq_tensor)
            yhat = output.argmax(1).item()

        word = next((w for w, idx in tokenizer.word_index.items() if idx == yhat), None)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return ' '.join(in_text.split()[1:])

# Handle the image before enter to the generate_caption function and print the caption and plot the image
def predict_one_image(image_path, model, tokenizer, max_length):
    # Load image
    img = Image.open(image_path).convert('RGB')

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Extract features from ResNet
    with torch.no_grad():
        feature = resnet(img_tensor).squeeze().cpu().numpy()

    # Generate caption
    caption = generate_caption(model, tokenizer, feature, max_length)

    # Display results
    print("Predicted Caption:", caption)
    plt.imshow(img)
    plt.axis('off')
    plt.title(caption)
    plt.show()

    return caption


from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_bleu_score(image_name, model, tokenizer, max_length, df, image_dir):
    # Get reference captions for the image
    references = df[df['image'] == image_name]['caption'].tolist()
    references = [ref.split() for ref in references]  # tokenize

    # Get predicted caption
    predicted = predict_one_image(os.path.join(image_dir, image_name), model, tokenizer, max_length)
    candidate = predicted.split()  # tokenize

    # BLEU calculation
    smoothie = SmoothingFunction().method4
    bleu1 = sentence_bleu(references, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu2 = sentence_bleu(references, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu4 = sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    print(f"BLEU-1: {bleu1:.4f}, BLEU-2: {bleu2:.4f}, BLEU-4: {bleu4:.4f}")
    return bleu1, bleu2, bleu4

image_name = "1026685415_0431cbf574.jpg"
image_dir = "/kaggle/input/flickr8k/Images"


bleu1, bleu2, bleu4 = calculate_bleu_score(image_name=image_name, model=model, tokenizer=tokenizer, max_length=max_length, df=df, image_dir=image_dir)

import pandas as pd
import os

# Download the data using kagglehub
path = kagglehub.dataset_download("adityajn105/flickr8k")
print("Path to dataset files:", path)

# Load the captions CSV file into a DataFrame
df = pd.read_csv('/kaggle/input/flickr8k/captions.txt')
df.head()  # Display first few rows of the captions file

# Define the path to the image directory
image_path = '/kaggle/input/flickr8k/Images'


df

import re

def text_preprocessing(data):
    # Convert captions to lowercase
    data['caption'] = data['caption'].apply(lambda x: x.lower())

    # Remove non-alphabetic characters
    data['caption'] = data['caption'].apply(lambda x: re.sub(r'[^a-z\s]', '', x))

    # Remove extra spaces
    data['caption'] = data['caption'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

    # Add startseq and endseq tokens to the caption
    data['caption'] = data['caption'].apply(lambda x: 'startseq ' + x + ' endseq')

    return data

# Apply preprocessing to the captions
df = text_preprocessing(df)
df.head()  # Check the first few rows after preprocessing




# Initialize ResNet-152 pre-trained model (without the final fully connected layer)
resnet_model = models.resnet152(pretrained=True)
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])  # Remove last FC layer
resnet_model.eval()  # Set model to evaluation mode

# Image transformations to match the ResNet input format
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Pre-trained ResNet normalization
])

# Function to extract features from an image
def extract_image_features(image_path, model, transform, device):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    with torch.no_grad():
        features = model(image)  # Extract features
    return features.squeeze().cpu().numpy()  # Convert features to numpy array

# Set device to CUDA (GPU) if available, else use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet_model.to(device)

# Extract features for all images and store them in a dictionary
features_dict = {}
image_files = [f for f in os.listdir(image_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
print(f"Total images to process: {len(image_files)}")

for img_file in tqdm(image_files):
    img_path = os.path.join(image_path, img_file)
    features_dict[img_file] = extract_image_features(img_path, resnet_model, transform, device)

# Save features to a file
with open('image_features_flickr8k.pkl', 'wb') as f:
    pickle.dump(features_dict, f)

print(f"Features extracted and saved for {len(features_dict)} images!")


# Load pre-trained VisionEncoderDecoderModel with ViT + BART
model_name = "nlpconnect/vit-gpt2-image-captioning"  # You can also use BART instead of GPT-2
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# Load the image processor and tokenizer (BART)
image_processor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()


def generate_caption(image_path, model, tokenizer, image_processor, max_length=16):
    # Load and process the image
    image = Image.open(image_path).convert('RGB')
    pixel_values = image_processor(image, return_tensors="pt").pixel_values

    # Generate caption using the model
    output_ids = model.generate(pixel_values, max_length=max_length, num_beams=4)

    # Decode the generated caption
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption


# Load saved image features
with open('image_features_flickr8k.pkl', 'rb') as f:
    features_dict = pickle.load(f)

# Generate captions for a subset of images (e.g., 100 images)
captions_dict = {}
sample_size = 100  # Process only the first 100 images
for img_file in tqdm(list(features_dict.keys())[:sample_size]):
    img_path = os.path.join(image_path, img_file)
    caption = generate_caption(img_path, model, tokenizer, image_processor)
    captions_dict[img_file] = caption

# Save captions to a file
with open('generated_captions_sample.pkl', 'wb') as f:
    pickle.dump(captions_dict, f)

print(f"Captions generated for {len(captions_dict)} images!")




def remove_tokens(caption):
    # Remove the 'startseq' and 'endseq' tokens
    caption = caption.replace('startseq ', '').replace(' endseq', '')
    return caption

def visualize_image_with_both_captions(image_file, captions_dict, original_captions, image_path):
    # Get the predicted caption for the image from captions_dict
    predicted_caption = captions_dict.get(image_file, "No predicted caption available")
    predicted_caption = remove_tokens(predicted_caption)  # Remove startseq and endseq from predicted caption

    # Get the original caption for the image from the DataFrame
    original_caption = original_captions.get(image_file, "No original caption available")
    original_caption = remove_tokens(original_caption)  # Remove startseq and endseq from original caption

    # Construct the image path
    img_path = os.path.join(image_path, image_file)

    # Open and display the image
    image = Image.open(img_path)
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis('off')  # Hide axes
    plt.title(f"Original Caption: {original_caption}\nPredicted Caption: {predicted_caption}")
    plt.show()

# Create a dictionary to map image files to their original captions
original_captions_dict = {row['image']: row['caption'] for _, row in df.iterrows()}

# Visualize some images with both original and predicted captions
sample_images = list(captions_dict.keys())[20:50]  # Take a subset of 10 images for visualization

for img_file in sample_images:
    visualize_image_with_both_captions(img_file, captions_dict, original_captions_dict, image_path)


# Load the pre-trained model
model_name = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
image_processor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Define the loss function (CrossEntropyLoss)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='mean')

# Function to calculate the loss for a batch of images and captions
def calculate_loss(batch_images, batch_captions, model, tokenizer, loss_fn, device):
    # Process the images and captions
    pixel_values = image_processor(batch_images, return_tensors="pt").pixel_values.to(device)
    target_ids = tokenizer(batch_captions, padding=True, return_tensors="pt").input_ids.to(device)

    # Forward pass: get model outputs
    outputs = model(pixel_values=pixel_values, labels=target_ids)

    # Get the loss
    loss = outputs.loss
    return loss.item()

# Example of how to compute loss for a sample batch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Assuming `batch_images` is a list of images and `batch_captions` is a list of ground truth captions (for a batch)
batch_images = [Image.open(os.path.join(image_path, img_file)).convert('RGB') for img_file in sample_images]
batch_captions = [captions_dict[img_file] for img_file in sample_images]

# Calculate the loss for the sample batch
loss = calculate_loss(batch_images, batch_captions, model, tokenizer, loss_fn, device)
print(f"Loss: {loss}")


# Function to calculate BLEU score between two captions
def calculate_bleu_score(reference_caption, candidate_caption):
    # Tokenize captions
    reference_tokens = [reference_caption.lower().split()]
    candidate_tokens = candidate_caption.lower().split()

    # Calculate BLEU score
    bleu_score = sentence_bleu(reference_tokens, candidate_tokens)

    return bleu_score

# List to store BLEU scores for each image
bleu_scores = []

# Iterate through each image and calculate BLEU score
for img_file in df['image']:
    # Get the original caption for the image
    reference_caption = df.loc[df['image'] == img_file, 'caption'].values[0]

    # Get the predicted caption for the image
    predicted_caption = captions_dict.get(img_file, None)

    # If predicted caption exists, calculate BLEU score
    if predicted_caption:
        bleu_score = calculate_bleu_score(reference_caption, predicted_caption)
        bleu_scores.append(bleu_score)

# Calculate the average BLEU score across all images
average_bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
print(f"Average BLEU Score: {average_bleu_score}")


# Download latest version
path = kagglehub.dataset_download("adityajn105/flickr8k")

print("Path to dataset files:", path)

dataset_path = '/kaggle/input/flickr8k'  # Base directory
print("Dataset files in the directory:")
for root, dirs, files in os.walk(dataset_path):
    print(root)
    print(f"  Subdirectories: {dirs}")
    print(f"  Files: {files}")


# Set paths for the flickr8k images

flickr_images_dir = '/kaggle/input/flickr8k/Images'  # Corrected path

# Check if the path exists
if os.path.exists(flickr_images_dir):
    print(f"Images directory exists: {flickr_images_dir}")
    print(f"Number of images: {len(os.listdir(flickr_images_dir))}")
else:
    print(f"Images directory not found: {flickr_images_dir}")



# Step 1: Extract features from images using a pre-trained CNN (ResNet-152)
def extract_image_features(image_dir, batch_size=32):
    """Extract features from images using a pre-trained CNN."""
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Load pre-trained ResNet-152 model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = models.resnet152(pretrained=True)
    # Remove the last FC layer to get features
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval()

    # Get list of image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Total images to process: {len(image_files)}")

    # Process images in batches
    features_dict = {}

    for i in tqdm(range(0, len(image_files), batch_size)):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        batch_file_names = []

        for img_file in batch_files:
            img_path = os.path.join(image_dir, img_file)
            try:
                image = Image.open(img_path).convert('RGB')
                image = transform(image)
                batch_images.append(image)
                batch_file_names.append(img_file)
            except Exception as e:
                print(f"Error processing {img_file}: {e}")

        # Skip empty batches
        if not batch_images:
            continue

        # Convert list to tensor
        batch_tensor = torch.stack(batch_images).to(device)

        # Extract features
        with torch.no_grad():
            batch_features = model(batch_tensor)
            batch_features = batch_features.squeeze().cpu().numpy()

        # Store features
        for idx, img_file in enumerate(batch_file_names):
            if len(batch_features.shape) == 1:  # Only one image in batch
                features_dict[img_file] = batch_features
            else:
                features_dict[img_file] = batch_features[idx]

    # Save features
    with open('image_features_flickr8k.pkl', 'wb') as f:
        pickle.dump(features_dict, f)

    print(f"Features extracted and saved for {len(features_dict)} images!")
    return features_dict


# Step 2: Generate captions using a pre-trained image captioning model
def generate_captions(image_dir):
    """Generate captions for images using a pre-trained model."""
    # Load pre-trained model and processors
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set generation parameters
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    # Get list of image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Total images to process: {len(image_files)}")

    # Generate captions
    captions_dict = {}

    for img_file in tqdm(image_files):
        img_path = os.path.join(image_dir, img_file)
        try:
            image = Image.open(img_path).convert('RGB')

            # Preprocess image
            pixel_values = image_processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)

            # Generate caption
            output_ids = model.generate(pixel_values, **gen_kwargs)

            # Decode caption
            caption = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            captions_dict[img_file] = caption

        except Exception as e:
            print(f"Error processing {img_file}: {e}")

    # Save captions
    with open('generated_captions_flickr8k.pkl', 'wb') as f:
        pickle.dump(captions_dict, f)

    print(f"Captions generated for {len(captions_dict)} images!")
    return captions_dict

# Step 3: Visualize some examples
def visualize_examples(image_dir, captions_dict, num_examples=10):
    """Visualize some examples of images and their generated captions."""
    # Get list of image files
    image_files = list(captions_dict.keys())

    # Select random images
    import random
    random.shuffle(image_files)
    selected_images = image_files[:num_examples]

    # Create a figure
    plt.figure(figsize=(15, 10))

    for i, img_file in enumerate(selected_images):
        # Load the image
        img_path = os.path.join(image_dir, img_file)
        image = Image.open(img_path).convert('RGB')

        # Get the caption
        caption = captions_dict[img_file]

        # Display image and caption
        plt.subplot(num_examples, 1, i+1)
        plt.imshow(image)
        plt.title(f"Image: {img_file}")
        plt.xlabel(f"Generated Caption: {caption}")
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()


def save_model(model, image_processor, tokenizer):
    """Save the trained model, image processor, and tokenizer."""
    try:
        model.save_pretrained("trained_vit_gpt2_flickr8k")
        image_processor.save_pretrained("trained_vit_gpt2_flickr8k")
        tokenizer.save_pretrained("trained_vit_gpt2_flickr8k")
        print("Model, image processor, and tokenizer saved!")
    except Exception as e:
        print(f"Error while saving the model: {e}")

# Step 1: Extract features from images
print("Extracting image features...")
features_dict = extract_image_features(flickr_images_dir)

# Step 2: Generate captions for images
print("Generating captions...")
captions_dict = generate_captions(flickr_images_dir)

# Step 3: Visualize some examples
print("Visualizing examples...")
visualize_examples(flickr_images_dir, captions_dict)


from flask import Flask, request, jsonify
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io
import base64
import threading
import nltk
from nltk.translate.bleu_score import sentence_bleu

# Download NLTK data for BLEU score calculation
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = Flask(__name__)

# Load the pre-trained models and processors
def load_models():
    # Load ViT-GPT2 model
    vit_gpt2_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    vit_image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    vit_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    # Load BLIP model
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move models to device
    vit_gpt2_model.to(device)
    blip_model.to(device)

    # Set generation parameters for ViT-GPT2
    vit_gpt2_model.config.decoder_start_token_id = vit_tokenizer.cls_token_id
    vit_gpt2_model.config.pad_token_id = vit_tokenizer.pad_token_id
    vit_gpt2_model.config.vocab_size = vit_gpt2_model.config.decoder.vocab_size

    return {
        'vit_gpt2': {
            'model': vit_gpt2_model,
            'processor': vit_image_processor,
            'tokenizer': vit_tokenizer
        },
        'blip': {
            'model': blip_model,
            'processor': blip_processor
        },
        'device': device
    }

# Generate caption using ViT-GPT2 model
def generate_vit_gpt2_caption(image, model_data):
    model = model_data['vit_gpt2']['model']
    processor = model_data['vit_gpt2']['processor']
    tokenizer = model_data['vit_gpt2']['tokenizer']
    device = model_data['device']

    # Set generation parameters
    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    # Preprocess image
    pixel_values = processor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Generate caption
    output_ids = model.generate(pixel_values, **gen_kwargs)

    # Decode caption
    caption = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    return caption

# Generate caption using BLIP model
def generate_blip_caption(image, model_data):
    model = model_data['blip']['model']
    processor = model_data['blip']['processor']
    device = model_data['device']

    # Preprocess image
    inputs = processor(image, return_tensors="pt").to(device)

    # Generate caption
    output = model.generate(**inputs, max_length=30)

    # Decode caption
    caption = processor.decode(output[0], skip_special_tokens=True)

    return caption

# Calculate BLEU score between two captions
def calculate_bleu_score(reference_caption, candidate_caption):
    # Tokenize captions
    reference_tokens = [reference_caption.lower().split()]
    candidate_tokens = candidate_caption.lower().split()

    # Calculate BLEU score
    bleu_score = sentence_bleu(reference_tokens, candidate_tokens)

    return bleu_score

# Load models at startup
model_data = load_models()
print(f"Models loaded on {model_data['device']}")

# API endpoint for ViT-GPT2 caption generation
@app.route('/predict_vit_gpt2', methods=['POST'])
def predict_vit_gpt2():
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400

    try:
        # Get the base64 encoded image
        image_data = request.json['image']

        # Decode the base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Generate caption from ViT-GPT2 model
        vit_gpt2_caption = generate_vit_gpt2_caption(image, model_data)

        return jsonify({
            'caption': vit_gpt2_caption
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API endpoint for BLIP caption generation
@app.route('/predict_blip', methods=['POST'])
def predict_blip():
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400

    try:
        # Get the base64 encoded image
        image_data = request.json['image']

        # Decode the base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Generate caption from BLIP model
        blip_caption = generate_blip_caption(image, model_data)

        return jsonify({
            'caption': blip_caption
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API endpoint for calculating BLEU score
@app.route('/calculate_bleu', methods=['POST'])
def calculate_bleu():
    if 'vit_caption' not in request.json or 'blip_caption' not in request.json:
        return jsonify({'error': 'Both captions are required'}), 400

    try:
        vit_caption = request.json['vit_caption']
        blip_caption = request.json['blip_caption']

        # Calculate BLEU scores
        vit_to_blip_bleu = calculate_bleu_score(blip_caption, vit_caption)
        blip_to_vit_bleu = calculate_bleu_score(vit_caption, blip_caption)

        return jsonify({
            'vit_to_blip_bleu': vit_to_blip_bleu,
            'blip_to_vit_bleu': blip_to_vit_bleu
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# HTML template for the web interface
@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dual Model Image Caption Generator</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                display: flex;
                flex-direction: column;
                align-items: center;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
            }
            .upload-section {
                margin: 20px 0;
                width: 100%;
            }
            .image-preview {
                margin: 20px 0;
                max-width: 500px;
                max-height: 500px;
                border-radius: 5px;
            }
            .caption-result {
                font-size: 18px;
                margin: 20px 0;
                padding: 15px;
                background-color: #f0f0f0;
                border-radius: 5px;
                width: 100%;
                box-sizing: border-box;
            }
            .model-caption {
                margin: 10px 0;
                padding: 10px;
                border-left: 4px solid #4CAF50;
            }
            .bleu-score {
                font-size: 14px;
                color: #666;
                margin-top: 5px;
            }
            .hidden {
                display: none;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin-top: 10px;
            }
            button:hover {
                background-color: #45a049;
            }
            input[type="file"] {
                margin: 10px 0;
            }
            .loading {
                border: 5px solid #f3f3f3;
                border-top: 5px solid #3498db;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                animation: spin 2s linear infinite;
                margin: 20px auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Dual Model Image Caption Generator</h1>
            <p>Upload an image to generate captions from two different models</p>

            <div class="upload-section">
                <input type="file" id="imageInput" accept="image/*">
                <button id="generateBtn">Generate Captions</button>
            </div>

            <div id="imagePreview" class="hidden">
                <h3>Image Preview:</h3>
                <img id="preview" class="image-preview">
            </div>

            <div id="loadingIndicator" class="hidden">
                <div class="loading"></div>
                <p>Generating captions...</p>
            </div>

            <div id="captionResult" class="caption-result hidden">
                <h3>Generated Captions:</h3>
                <div class="model-caption">
                    <strong>ViT-GPT2 Model:</strong>
                    <p id="vit-gpt2-caption">Loading...</p>
                    <div class="bleu-score">BLEU Score: <span id="vit-bleu-score">-</span></div>
                </div>
                <div class="model-caption">
                    <strong>BLIP Model:</strong>
                    <p id="blip-caption">Loading...</p>
                    <div class="bleu-score">BLEU Score: <span id="blip-bleu-score">-</span></div>
                </div>
            </div>
        </div>

        <script>
            // Preview the selected image
            document.getElementById('imageInput').addEventListener('change', function(event) {
                const file = event.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        document.getElementById('preview').src = e.target.result;
                        document.getElementById('imagePreview').classList.remove('hidden');
                        document.getElementById('captionResult').classList.add('hidden');
                    };
                    reader.readAsDataURL(file);
                }
            });

            // Generate captions when button is clicked
            document.getElementById('generateBtn').addEventListener('click', function() {
                const fileInput = document.getElementById('imageInput');
                if (fileInput.files.length === 0) {
                    alert('Please select an image first');
                    return;
                }

                const file = fileInput.files[0];
                const reader = new FileReader();

                document.getElementById('loadingIndicator').classList.remove('hidden');
                document.getElementById('captionResult').classList.add('hidden');

                reader.onload = function(e) {
                    const base64Image = e.target.result.split(',')[1];

                    // Show caption result area with loading indicators
                    document.getElementById('captionResult').classList.remove('hidden');
                    document.getElementById('vit-gpt2-caption').textContent = "Loading...";
                    document.getElementById('blip-caption').textContent = "Loading...";
                    document.getElementById('vit-bleu-score').textContent = "-";
                    document.getElementById('blip-bleu-score').textContent = "-";

                    let vitCaption = '';
                    let blipCaption = '';

                    // Call ViT-GPT2 endpoint
                    fetch('/predict_vit_gpt2', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            image: base64Image
                        })
                    })
                    .then(response => response.json())
                    .then(data => {
                        vitCaption = data.caption;
                        document.getElementById('vit-gpt2-caption').textContent = vitCaption;

                        // If both captions are ready, calculate BLEU scores
                        if (blipCaption) {
                            calculateBleuScores(vitCaption, blipCaption);
                        }
                    })
                    .catch(error => {
                        document.getElementById('vit-gpt2-caption').textContent = "Error: " + error;
                    });

                    // Call BLIP endpoint
                    fetch('/predict_blip', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            image: base64Image
                        })
                    })
                    .then(response => response.json())
                    .then(data => {
                        blipCaption = data.caption;
                        document.getElementById('blip-caption').textContent = blipCaption;
                        document.getElementById('loadingIndicator').classList.add('hidden');

                        // If both captions are ready, calculate BLEU scores
                        if (vitCaption) {
                            calculateBleuScores(vitCaption, blipCaption);
                        }
                    })
                    .catch(error => {
                        document.getElementById('blip-caption').textContent = "Error: " + error;
                        document.getElementById('loadingIndicator').classList.add('hidden');
                    });
                };

                reader.readAsDataURL(file);
            });

            // Calculate BLEU scores
            function calculateBleuScores(vitCaption, blipCaption) {
                fetch('/calculate_bleu', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        vit_caption: vitCaption,
                        blip_caption: blipCaption
                    })
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('vit-bleu-score').textContent = data.vit_to_blip_bleu.toFixed(4);
                    document.getElementById('blip-bleu-score').textContent = data.blip_to_vit_bleu.toFixed(4);
                })
                .catch(error => {
                    console.error("Error calculating BLEU scores:", error);
                });
            }
        </script>
    </body>
    </html>
    """

# For Google Colab, we'll use Colab's built-in method to expose the app
def run_in_colab():
    from google.colab.output import eval_js
    print("Starting Flask server...")

    # Start the Flask app in a separate thread
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False)).start()

    # Use JavaScript to create a proxy
    js = """
    async function setupProxy() {
      const url = await google.colab.kernel.proxyPort(8000);
      const iframe = document.createElement('iframe');
      iframe.src = url + '?v=' + new Date().getTime();  // Add timestamp to prevent caching
      iframe.width = '100%';
      iframe.height = '800px';
      iframe.frameBorder = 0;
      document.body.appendChild(iframe);
      return url;
    }
    setupProxy()
    """
    public_url = eval_js(js)
    print(f"Flask app running at: {public_url}")

# Run the Flask app
if __name__ == '__main__':
    try:
        # Check if running in Colab
        from google.colab import output
        run_in_colab()
    except ImportError:
        # If not in Colab, run normally
        app.run(debug=True, host='0.0.0.0', port=8000)
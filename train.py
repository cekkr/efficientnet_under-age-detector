import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v2
from PIL import Image
import requests
import random
import io
import base64
from tqdm import tqdm
import time

from model import *

device_name = "mps" # or cuda

class SDWebUIAPI:
    def __init__(self, url="http://127.0.0.1:7860"):
        self.url = url

    def generate_image(self, prompt, negative_prompt=""):
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": 20,
            "width": 512,
            "height": 512,
            "cfg_scale": 7
        }

        response = requests.post(url=f'{self.url}/sdapi/v1/txt2img', json=payload)
        r = response.json()

        # Decodifica l'immagine da base64
        image = Image.open(io.BytesIO(base64.b64decode(r['images'][0])))
        return image


class DynamicAgeDataset(Dataset):
    def __init__(self, sd_api, samples_per_epoch=1000, transform=None):
        self.sd_api = sd_api
        self.samples_per_epoch = samples_per_epoch
        self.transform = transform
        self.cache = []

        # Template per i prompt
        self.locations = [
            "in a park", "at the beach", "in a city street", "in a library",
            "at a cafe", "in a shopping mall", "at home", "in a museum",
            "at a restaurant", "in a garden"
        ]

        self.clothing = [
            "casual clothes", "t-shirt and jeans", "summer dress",
            "formal attire", "sportswear", "winter coat", "school uniform",
            "hoodie and sneakers", "shorts and tank top", "down jacket", "puffer jacket", "shiny down jacket"
        ]

        self.styles = [
            "with their parents", "with their friends", "with his granpa and grandma"
        ]

    def generate_prompt(self, age):
        location = random.choice(self.locations)
        clothing = random.choice(self.clothing)
        style = random.choice(self.styles)
        sex = random.choice(["male", "female"])

        prompt = f"{age} year old {sex} {location}, wearing {clothing}, {style}"

        # Prompt negativo standard per qualità e sicurezza
        negative_prompt = "nsfw, nude, poorly rendered, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"

        return prompt, negative_prompt

    def generate_batch(self):
        print("Generating new batch of images...")
        self.cache = []

        for _ in tqdm(range(self.samples_per_epoch)):
            # Genera età casuale (distribuzione uniforme tra 8 e 25)
            age = random.randint(1, 40)
            is_minor = 0
            if age < 12:
                is_minor = 1
            elif age > 18:
                is_minor = 0
            else:
                is_minor = 1-((age - 12)/6)

            prompt, negative_prompt = self.generate_prompt(age)

            try:
                image = self.sd_api.generate_image(prompt, negative_prompt)
                if self.transform:
                    image = self.transform(image)

                self.cache.append((image, is_minor))

                # Attende brevemente per non sovraccaricare l'API
                time.sleep(0.1)

            except Exception as e:
                print(f"Error generating image: {e}")
                continue

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        if len(self.cache) <= idx:
            self.generate_batch()

        image, label = self.cache[idx]
        return image, torch.tensor(label, dtype=torch.float32)


import os
import json
import hashlib
from pathlib import Path


class DynamicAgeDataset(Dataset):
    def __init__(self, sd_api, samples_per_epoch=1000, transform=None, cache_dir="dataset_cache"):
        self.sd_api = sd_api
        self.samples_per_epoch = samples_per_epoch
        self.transform = transform
        self.cache = []

        # Setup cache directory
        self.cache_dir = Path(cache_dir)
        self.images_dir = self.cache_dir / "images"
        self.metadata_file = self.cache_dir / "metadata.json"
        self._setup_cache_dirs()

        # Template per i prompt (mantenuti dal tuo codice)
        self.locations = [
            "in a park", "at the beach", "in a city street", "in a library",
            "at a cafe", "in a shopping mall", "at home", "in a museum",
            "at a restaurant", "in a garden"
        ]

        self.clothing = [
            "casual clothes", "t-shirt and jeans", "summer dress",
            "formal attire", "sportswear", "winter coat", "school uniform",
            "hoodie and sneakers", "shorts and tank top", "down jacket",
            "puffer jacket", "shiny down jacket"
        ]

        self.styles = [
            "with their parents", "with their friends", "alone"
        ]

        # Carica il metadata esistente o inizializza nuovo
        self.metadata = self._load_metadata()

    def _setup_cache_dirs(self):
        """Crea le directory necessarie per il caching"""
        self.cache_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)

    def _load_metadata(self):
        """Carica il metadata esistente o crea nuovo file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"images": {}}

    def _save_metadata(self):
        """Salva il metadata su disco"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _generate_image_hash(self, prompt, negative_prompt):
        """Genera un hash univoco per l'immagine basato sui prompt"""
        content = f"{prompt}|{negative_prompt}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_image_path(self, image_hash):
        """Restituisce il path completo dell'immagine"""
        return self.images_dir / f"{image_hash}.png"

    def generate_prompt(self, age):
        location = random.choice(self.locations)
        clothing = random.choice(self.clothing)
        style = random.choice(self.styles)
        sex = random.choice(["male", "female"])

        prompt = f"{age} year old {sex} {location}, wearing {clothing}, {style}"
        negative_prompt = "nsfw, nude, poorly rendered, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"

        return prompt, negative_prompt

    def _calculate_minor_score(self, age):
        """Calcola lo score di minorità basato sull'età"""
        if age < 12:
            return 1.0
        elif age > 18:
            return 0.0
        else:
            return 1 - ((age - 12) / 6)

    def generate_batch(self):
        print("Checking cache and generating new images as needed...")
        self.cache = []
        needed_samples = self.samples_per_epoch - len(self.metadata["images"])

        if needed_samples > 0:
            print(f"Generating {needed_samples} new images...")
            for _ in tqdm(range(needed_samples)):
                age = random.randint(1, 30)
                is_minor = self._calculate_minor_score(age)
                prompt, negative_prompt = self.generate_prompt(age)
                image_hash = self._generate_image_hash(prompt, negative_prompt)

                if image_hash not in self.metadata["images"]:
                    try:
                        image = self.sd_api.generate_image(prompt, negative_prompt)
                        image_path = self._get_image_path(image_hash)
                        image.save(image_path)

                        self.metadata["images"][image_hash] = {
                            "age": age,
                            "is_minor": is_minor,
                            "prompt": prompt,
                            "negative_prompt": negative_prompt
                        }

                        self._save_metadata()
                        time.sleep(0.1)

                    except Exception as e:
                        print(f"Error generating image: {e}")
                        continue

        # Carica le immagini esistenti nella cache
        image_hashes = list(self.metadata["images"].keys())
        selected_hashes = random.sample(image_hashes, min(self.samples_per_epoch, len(image_hashes)))

        for image_hash in selected_hashes:
            image_path = self._get_image_path(image_hash)
            try:
                image = Image.open(image_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)

                is_minor = self.metadata["images"][image_hash]["is_minor"]
                self.cache.append((image, is_minor))

            except Exception as e:
                print(f"Error loading cached image {image_hash}: {e}")
                continue

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        if len(self.cache) <= idx:
            self.generate_batch()

        image, label = self.cache[idx]
        return image, torch.tensor(label, dtype=torch.float32)

    def get_stats(self):
        """Restituisce statistiche sul dataset cachato"""
        total_images = len(self.metadata["images"])
        ages = [data["age"] for data in self.metadata["images"].values()]
        minor_scores = [data["is_minor"] for data in self.metadata["images"].values()]

        return {
            "total_images": total_images,
            "avg_age": sum(ages) / len(ages) if ages else 0,
            "avg_minor_score": sum(minor_scores) / len(minor_scores) if minor_scores else 0,
            "cache_size_mb": sum(os.path.getsize(f) for f in self.images_dir.glob("*.png")) / (1024 * 1024)
        }

def train_model(model, train_dataset, val_dataset, num_epochs=10, batch_size=32):
    global device_name
    device = torch.device(device_name)
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total

        # Validazione
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total

        print(f'Training Loss: {running_loss / len(train_loader):.4f}')
        print(f'Training Accuracy: {train_accuracy:.2f}%')
        print(f'Validation Loss: {val_loss / len(val_loader):.4f}')
        print(f'Validation Accuracy: {val_accuracy:.2f}%')
        print('-' * 60)

        # Genera nuovo batch per il prossimo epoch
        train_dataset.generate_batch()
        val_dataset.generate_batch()


def save_model(model, base_path):
    model.eval()
    # Salva solo lo state_dict per il modello completo
    torch.save(model.state_dict(), f'{base_path}_full.pth')

    # Per l'inferenza ottimizzata, usa script_module
    scripted_model = torch.jit.script(model)
    scripted_model.save(f'{base_path}_scripted.pt')


def main():
    # Inizializza l'API di Stable Diffusion WebUI
    sd_api = SDWebUIAPI()

    # Trasformazioni per le immagini
    transform = transforms.Compose([
        transforms.Resize(512),  # Ridimensiona mantenendo l'aspect ratio
        transforms.CenterCrop(512),  # Ritaglia al centro per ottenere un quadrato
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Media ImageNet
            std=[0.229, 0.224, 0.225]  # Deviazione standard ImageNet
        )
    ])

    # Crea i dataset
    # Usa il caching
    train_dataset = DynamicAgeDataset(
        sd_api,
        samples_per_epoch=1000,
        transform=transform,
        cache_dir="train_cache"
    )

    val_dataset = DynamicAgeDataset(
        sd_api,
        samples_per_epoch=100,
        transform=transform,
        cache_dir="val_cache"
    )
54rew8rfdacl56
    # Controlla le statistiche5
    model.eval()
    example = torch.rand(1, 3, 512, 512).to(device_name)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save('age_detector_sd_optimized.pt')


if __name__ == '__main__':
    main()
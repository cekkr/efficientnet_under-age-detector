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

from model_efficient import *

device_name = "cuda" # or cuda
learning_rate = 0.00001

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



import os
import json
import hashlib
from pathlib import Path

def debug_tensor_conversion(image_path):
    os.makedirs('debug', exist_ok=True)
    
    # Carica e salva originale
    original = Image.open(image_path).convert('RGB')
    original.save('debug/0_original.png')
    
    # ToTensor con controlli intermedi
    tensor = transforms.ToTensor()(original)
    print("Post ToTensor:", tensor.dtype, tensor.min(), tensor.max())
    pil_check1 = transforms.ToPILImage()(tensor.clamp(0, 1))  # Clamp per sicurezza
    pil_check1.save('debug/1_post_tensor.png')
    
    # Resize
    resized_tensor = transforms.Resize(600)(tensor)
    print("Post Resize:", resized_tensor.dtype, resized_tensor.min(), resized_tensor.max())
    pil_check2 = transforms.ToPILImage()(resized_tensor.clamp(0, 1))
    pil_check2.save('debug/2_post_resize.png')
    
    # CenterCrop
    cropped_tensor = transforms.CenterCrop(600)(resized_tensor)
    print("Post Crop:", cropped_tensor.dtype, cropped_tensor.min(), cropped_tensor.max())
    pil_check3 = transforms.ToPILImage()(cropped_tensor.clamp(0, 1))
    pil_check3.save('debug/3_post_crop.png')
    
    # Normalize
    normalized = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )(cropped_tensor)
    print("Post Normalize:", normalized.dtype, normalized.min(), normalized.max())
    
    return normalized

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
            "down jacket", "swimming suit"
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

    '''
    def __getitem__(self, idx):
        if len(self.cache) <= idx:
            self.generate_batch()

        image, label = self.cache[idx]
        return image, torch.tensor(label, dtype=torch.float32)
    '''

    def __getitem__(self, idx):

        if len(self.cache) <= idx:
            self.generate_batch()

        image_path = self._get_image_path(list(self.metadata["images"].keys())[idx])
        #debug_tensor_conversion(image_path)  # Add this line

        try:
            # Debug: salva l'immagine originale prima di qualsiasi trasformazione
            original = Image.open(image_path).convert('RGB')

            if False:
                original.save('debug/0_pre_transform.png')
            
            if self.transform:
                image = self.transform(original)
            else:
                image = transforms.ToTensor()(original)

            is_minor = self.metadata["images"][list(self.metadata["images"].keys())[idx]]["is_minor"]
            return image, torch.tensor(is_minor, dtype=torch.float32)

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Fallback a immagine casuale dal dataset
            return self.__getitem__(random.randint(0, len(self) - 1))

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


def visualize_transform_steps(image_tensor):
   os.makedirs('debug', exist_ok=True)
   
   # Step 1: Original tensor to PIL
   original = transforms.ToPILImage()(image_tensor)
   original.save('debug/1_original.png')
   
   # Step 2: Resize
   resized = transforms.Resize(600)(original) 
   resized.save('debug/2_resized.png')
   
   # Step 3: CenterCrop
   cropped = transforms.CenterCrop(600)(resized)
   cropped.save('debug/3_cropped.png')
   
   # Step 4: ToTensor
   tensor = transforms.ToTensor()(cropped)
   tensor_img = transforms.ToPILImage()(tensor)
   tensor_img.save('debug/4_to_tensor.png')
   
   # Step 5: Normalize
   normalized = transforms.Normalize(
       mean=[0.485, 0.456, 0.406], 
       std=[0.229, 0.224, 0.225]
   )(tensor)
   
   # Denormalize
   denorm = transforms.Normalize(
       mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
       std=[1/0.229, 1/0.224, 1/0.225]
   )(normalized)
   
   denorm_img = transforms.ToPILImage()(denorm)
   denorm_img.save('debug/5_normalized.png')
   
   # Print tensor stats
   print(f"Original tensor - Min: {image_tensor.min():.3f}, Max: {image_tensor.max():.3f}")
   print(f"Final tensor - Min: {normalized.min():.3f}, Max: {normalized.max():.3f}")

def visualize_transforms(image_tensor, transformed_tensor):
    os.makedirs('debug', exist_ok=True)
    
    # Per l'immagine originale
    original_image = transforms.ToPILImage()(image_tensor)
    original_image.save('debug/original.png')
    
    # Denormalizziamo prima di convertire in PIL
    denorm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )(transformed_tensor)
    
    transformed_image = transforms.ToPILImage()(denorm)
    transformed_image.save('debug/transformed.png')
    
    print(f"Original tensor stats - Min: {image_tensor.min():.3f}, Max: {image_tensor.max():.3f}")
    print(f"Transformed tensor stats - Min: {transformed_tensor.min():.3f}, Max: {transformed_tensor.max():.3f}")

def train_model(model, train_dataset, val_dataset, num_epochs=10, batch_size=32):
    global device_name
    device = torch.device(device_name)
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        if False:
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                if batch_idx == 0:  # Only for first batch
                    visualize_transforms(inputs[0], inputs[0])
                break

        for inputs, labels in tqdm(train_loader):    
            # Get original image from dataset
            if False:
                original_image = train_dataset.cache[0][0]  # Get first image
                visualize_transforms(original_image, inputs[0])
            
            # Debug tensori
            if False:
                print(f"Input tensor range: {inputs.min():.3f} to {inputs.max():.3f}")
                print(f"Input tensor mean: {inputs.mean():.3f}")
                print(f"Input tensor std: {inputs.std():.3f}")

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

def save_checkpoint(model, optimizer, epoch, loss, accuracy, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']

def train_model(model, train_dataset, val_dataset, num_epochs=10, batch_size=32, checkpoint_dir='checkpoints'):
    global device_name
    global learning_rate

    device = torch.device(device_name)
    model = model.to(device)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        start_epoch, last_loss, last_accuracy = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Resuming from epoch {start_epoch}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        
        # Training loop
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
        
        # Validation loop
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
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, 
                optimizer, 
                epoch + 1, 
                running_loss / len(train_loader),
                train_accuracy,
                checkpoint_path
            )
            print(f"Checkpoint saved at epoch {epoch + 1}")
        
        # Generate new batch for next epoch
        train_dataset.generate_batch()
        val_dataset.generate_batch()
    
    # Save final model
    save_model(model, 'model/age_detector')

def main():
    # Inizializza l'API di Stable Diffusion WebUI
    sd_api = SDWebUIAPI()

    # Trasformazioni per le immagini
    transform = transforms.Compose([
        transforms.Resize(600),
        transforms.CenterCrop(600),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Crea i dataset
    # Usa il caching
    train_dataset = DynamicAgeDataset(
        sd_api,
        samples_per_epoch=2000,
        transform=transform,
        cache_dir="train_cache"
    )

    val_dataset = DynamicAgeDataset(
        sd_api,
        samples_per_epoch=200,
        transform=transform,
        cache_dir="val_cache"
    )

    # Controlla le statistiche
    print("Training dataset stats:", train_dataset.get_stats())
    print("Validation dataset stats:", val_dataset.get_stats())

    # Inizializza e addestra il modello
    model = AgeDetector()
    train_model(model, train_dataset, val_dataset, num_epochs=180, batch_size=8)

    # Salva il modello
    save_model(model, 'model/age_detector')
    #torch.save(model.state_dict(), 'age_detector_sd.pth')

    # Esporta in TorchScript per l'inferenza ottimizzata
    model.eval()
    example = torch.rand(1, 3, 512, 512).to(device_name)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save('model/age_detector_sd_optimized.pt')


if __name__ == '__main__':
    main()
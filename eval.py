import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from PIL import Image
import argparse
import sys
import warnings

from model_efficient import *

warnings.filterwarnings("ignore")

class AgePredictor:
    def __init__(self, model_path):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Carica il modello
            if model_path.endswith('.pth'):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model = AgeDetector()
                self.model.load_state_dict(checkpoint)
            elif model_path.endswith('.pt'):
                self.model = torch.jit.load(model_path)
            else:
                raise ValueError("Il formato del modello deve essere .pth o .pt")

            self.model.to(self.device)
            self.model.eval()

            self.transform = transforms.Compose([
                transforms.Resize(600),
                transforms.CenterCrop(600),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        except Exception as e:
            print(f"Errore nel caricamento del modello: {str(e)}")
            sys.exit(1)

    def predict(self, image_path, threshold=0.5):
        try:
            # Carica e preprocessa l'immagine
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)

            # Predizione
            with torch.no_grad():
                output = self.model(image_tensor).squeeze()
                probability = output.item()
                is_minor = probability > threshold

            return {
                'is_minor': is_minor,
                'probability': probability,
                'confidence': abs(probability - 0.5) * 2  # Converte in percentuale di confidenza
            }

        except FileNotFoundError:
            print(f"Errore: Il file {image_path} non esiste")
            return None
        except Exception as e:
            print(f"Errore durante la predizione: {str(e)}")
            return None


def main():
    parser = argparse.ArgumentParser(description='Predici se un soggetto è minorenne da un\'immagine')
    parser.add_argument('model_path', type=str, help='Percorso del modello (.pth o .pt)')
    parser.add_argument('image_path', type=str, help='Percorso dell\'immagine da analizzare')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Soglia di confidenza (default: 0.5)')

    args = parser.parse_args()

    # Inizializza il predictor
    predictor = AgePredictor(args.model_path)

    # Esegui la predizione
    result = predictor.predict(args.image_path, args.threshold)

    if result:
        print("\nAnalysis result:")
        print(f"It's underage: {'Yes' if result['is_minor'] else 'No'}")
        print(f"Probability: {result['probability']:.3f}")
        print(f"Confidence: {result['confidence'] * 100:.1f}%")


if __name__ == "__main__":
    main()
import os
import json
import subprocess
import sys

with open('test_images/metadata.json') as f:
    metadata = json.load(f)

model_path = 'model/age_detector_full.pth'
eval_script = 'eval.py'

for filename in os.listdir('test_images'):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_id = os.path.splitext(filename)[0]
        
        image_data = metadata['images'].get(image_id)
        if not image_data:
            continue

        print(f"\nAnalyzing {filename}")
        print(f"Original metadata:")
        print(f"Age: {image_data['age']}")
        print(f"Is minor: {image_data['is_minor']}")
        
        cmd = [sys.executable, eval_script, model_path, f'test_images/{filename}', '--threshold', '0.5']
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("\nModel evaluation:")
        print(result.stdout)

        print("================================================\n")
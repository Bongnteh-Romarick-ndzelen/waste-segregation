# inference_test.py
# Test your trained model on any image

import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import os

# Load model
model = models.mobilenet_v2()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 6)
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))  # or your saved .pth
model.eval()

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def classify_image(image_path):
    if not os.path.exists(image_path):
        print("Image not found!")
        return

    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    label = class_names[predicted.item()]
    conf = confidence.item()

    print(f"Predicted: {label.upper()}")
    print(f"Confidence: {conf:.2%}")
    print(f"All probabilities:")
    for name, prob in zip(class_names, probabilities[0]):
        print(f"  {name}: {prob.item():.2%}")

# Example usage
if __name__ == "__main__":
    # Put your test images in a folder called test_images/
    classify_image("test_images/plastic_bottle.jpg")
    classify_image("test_images/cardboard_box.jpg")
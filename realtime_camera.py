# realtime_camera.py
# Real-time waste classification using webcam

import cv2
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import time

# Load model
model = models.mobilenet_v2()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 6)
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
model.eval()

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
colors = {
    'cardboard': (255, 191, 0),    # Amber
    'glass': (0, 255, 0),         # Green
    'metal': (192, 192, 192),     # Silver
    'paper': (173, 216, 230),     # Light Blue
    'plastic': (255, 255, 0),     # Yellow
    'trash': (255, 0, 0)          # Red
}

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to PIL
    pil_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(pil_img)

    # Preprocess
    input_tensor = transform(pil_img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    label = class_names[predicted.item()]
    conf = confidence.item()
    color = colors.get(label, (255, 255, 255))

    # Display
    text = f"{label.upper()} ({conf:.1%})"
    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    cv2.imshow('Waste Classifier - Live Demo', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
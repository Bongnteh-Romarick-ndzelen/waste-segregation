# jetson_deploy.py
# For deployment on Jetson Nano: camera + serial to Arduino

import cv2
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import serial
import time

# Serial to Arduino (adjust port)
try:
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    time.sleep(2)  # Wait for Arduino reset
    print("Connected to Arduino")
except:
    print("Arduino not found! Running in simulation mode.")
    ser = None

# Load model (optimized for Jetson)
model = models.mobilenet_v2()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 6)
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
model.eval()

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(0)
print("Jetson Waste Segregator Running...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    pil_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(pil_img)
    input_tensor = transform(pil_img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    label = class_names[predicted.item()]
    conf = confidence.item()

    # Send command to Arduino if confident
    if conf > 0.90:
        command = f"SORT_{label.upper()}\n"
        if ser:
            ser.write(command.encode())
            print(f"Sent: {command.strip()}")
        else:
            print(f"[SIM] {command.strip()}")
    else:
        print("Low confidence → Trash")
        if ser:
            ser.write(b"SORT_TRASH\n")

    # Display
    cv2.putText(frame, f"{label} ({conf:.1%})", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow('Jetson Waste Segregator', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()
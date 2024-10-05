import cv2
import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# Define your model architecture (make sure it matches the training architecture)
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = torch.nn.Linear(32 * 56 * 56, 128)
        self.fc2 = torch.nn.Linear(128, 3)  # 3 classes: spitting, graffiti, no graffiti

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load your trained model
model = SimpleNet()  # Create an instance of your model
model.load_state_dict(torch.load('classification_model.pth'))  # Load the state dict
model.eval()  # Set the model to evaluation mode

# Initialize video capture
cap = cv2.VideoCapture(0)

# Create a directory to save screenshots
output_dir = 'screenshots'
os.makedirs(output_dir, exist_ok=True)

# Initialize a counter for naming the screenshots
screenshot_counter = 0

# Set up data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Variables for video recording
is_recording = False
recording_filename = "recording.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
fps = 20.0
frame_size = (640, 480)  # Adjust to your camera's resolution
out = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(rgb_frame)  # Convert frame to PIL Image
    img_tensor = transform(img_pil).unsqueeze(0)  # Apply transformations and add batch dimension

    # Perform inference
    with torch.no_grad():
        outputs = model(img_tensor)  # Forward pass
        _, predicted = torch.max(outputs.data, 1)  # Get the predicted class

    # Draw the predicted class on the frame
    label = predicted.item()  # Get the label as an integer
    label_text = f'Class: {label}'
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Start recording when class 0 (spitting) or class 1 (graffiti) is detected
    if label in [0, 1]:
        if not is_recording:
            out = cv2.VideoWriter(recording_filename, fourcc, fps, frame_size)  # Start recording
            is_recording = True
            print("Recording started.")
        
        # Write the frame to the video file
        out.write(frame)
        
        # Display recording indicator on the frame
        cv2.putText(frame, 'Recording...', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # Stop recording if it was in progress
        if is_recording:
            out.release()  # Stop recording
            is_recording = False
            print("Recording stopped.")

    cv2.imshow('Vandalism Detection', frame)

    # If a specific class is detected (you can modify this as needed), save the frame
    if label in [1]:  # Adjust the conditions based on your class labels
        screenshot_filename = os.path.join(output_dir, f'screenshot_{screenshot_counter}.jpg')
        cv2.imwrite(screenshot_filename, frame)
        screenshot_counter += 1
        print(f"Saved screenshot: {screenshot_filename}")

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
if out is not None:
    out.release()  # Ensure the video writer is released
cv2.destroyAllWindows()

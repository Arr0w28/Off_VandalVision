# Server B: Process video stream from Server A and perform inference

import cv2
import torch
import requests
import numpy as np
from PIL import Image
from flask import Flask, Response
from torchvision import transforms
from io import BytesIO

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

# Set up data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Flask app to serve the processed video stream
app = Flask(__name__)

# URL of the stream from Server A
stream_url = 'http://172.168.71.103:5003/video_feed'

@app.route('/process_video')
def process_video():
    def generate_frames():
        # Stream the video feed from Server A
        stream = requests.get(stream_url, stream=True)
        if stream.status_code != 200:
            return "Error fetching video stream"

        byte_data = b''  # To hold the byte data coming in chunks
        for chunk in stream.iter_content(chunk_size=1024):
            byte_data += chunk
            a = byte_data.find(b'\xff\xd8')  # JPEG start
            b = byte_data.find(b'\xff\xd9')  # JPEG end
            if a != -1 and b != -1:
                # Extract the JPEG image
                jpg = byte_data[a:b + 2]
                byte_data = byte_data[b + 2:]
                img_np = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

                # Convert the frame to PIL Image and then to tensor
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(rgb_frame)
                img_tensor = transform(img_pil).unsqueeze(0)

                # Perform inference
                with torch.no_grad():
                    outputs = model(img_tensor)  # Forward pass
                    _, predicted = torch.max(outputs.data, 1)  # Get the predicted class

                # Draw the predicted class on the frame
                label = predicted.item()  # Get the label as an integer
                label_text = f'Class: {label}'
                cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Encode the frame back to JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                # Yield the frame over HTTP as byte chunks
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)  # Running on localhost at port 5001
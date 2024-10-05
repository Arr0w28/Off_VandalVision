import os
import torch
import numpy as np
import cv2
import RRDBNet_arch as arch
from flask import Flask, request, jsonify, Response
from io import BytesIO
from PIL import Image
import requests

app = Flask(__name__)

# Set the model path and device
model_path = 'models/RRDB_PSNR_x4.pth'  # Path to your trained model
device = torch.device('cpu')  # Using CPU (adjust if needed)

# Initialize and load the model
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
model.eval()  # Set model to evaluation mode
model = model.to(device)

print(f'Model loaded from {model_path}.')

# Ensure the results directory exists
results_dir = 'results/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Function to process the image using the model
def super_resolve_image(image):
    # Convert PIL image to numpy array
    img_np = np.array(image).astype(np.float32) / 255.0
    img_np = torch.from_numpy(np.transpose(img_np[:, :, [2, 1, 0]], (2, 0, 1))).float()  # Change to RGB, (C, H, W)
    img_LR = img_np.unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Run inference with the model
    with torch.no_grad():
        output = model(img_LR).squeeze(0).cpu().numpy()

    # Convert back to [0, 255] and BGR order for OpenCV
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # Convert from RGB to BGR
    output = (output * 255.0).clip(0, 255).astype(np.uint8)  # Clip the values

    return output

def send_frame(frame):
    """Send the processed frame to another server."""
    _, buffer = cv2.imencode('.jpg', frame)
    frame_data = buffer.tobytes()
    
    # Send the frame as multipart/form-data
    destination_url = 'http://localhost:5003/receive_frame'  # Change this to your desired URL
    response = requests.post(destination_url, files={'file': ('frame.jpg', frame_data, 'image/jpeg')})
    return response

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Read image and apply super-resolution
    img = Image.open(BytesIO(file.read()))
    output_img = super_resolve_image(img)

    # Save the processed image
    result_path = os.path.join(results_dir, f'{file.filename}_rlt.png')
    cv2.imwrite(result_path, output_img)
    print(f'Saved result to {result_path}')

    return jsonify({"message": f"Image processed and saved to {result_path}"}), 200

@app.route('/process_video', methods=['POST'])
def process_video():
    """Process incoming video frames and send them to another server."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Read image and apply super-resolution
    img = Image.open(BytesIO(file.read()))
    output_img = super_resolve_image(img)

    # Send the processed frame to the destination server
    send_frame(output_img)

    return jsonify({"message": "Frame processed and sent to the destination server"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)  # Running on port 5002

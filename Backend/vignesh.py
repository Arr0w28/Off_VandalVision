import os
import cv2
import numpy as np
from flask import Flask, request, Response
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import requests

app = Flask(__name__)

# Set the YOLOv8 model path
yolo_model_path = 'yolov8m.pt'  # Path to your YOLOv8m model
yolo_model = YOLO(yolo_model_path)  # Load YOLOv8 model
print(f'Model loaded from {yolo_model_path}.')

# Tiling settings
tile_size = 128  # Adjust this based on your memory limit
tile_overlap = 32  # Overlap to avoid seams between tiles

# Streamer server details (adjust accordingly)
streamer_url = 'http://:5004/stream_video'  # Endpoint of the streamer server

# Helper function to split image into tiles
def split_image_into_tiles(img, tile_size, overlap):
    h, w, _ = img.shape
    tiles = []
    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            x_end = min(x + tile_size, w)
            y_end = min(y + tile_size, h)
            tile = img[y:y_end, x:x_end]
            tiles.append((x, y, tile))
    return tiles, h, w

# Helper function to stitch tiles back together
def stitch_tiles(tiles, h, w, tile_size, overlap):
    output_img = np.zeros((h, w, 3), dtype=np.float32)
    for (x, y, tile) in tiles:
        x_end = min(x + tile.shape[1], w)
        y_end = min(y + tile.shape[0], h)
        output_img[y:y_end, x:x_end] = tile
    return output_img

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'files' not in request.files:
        return {"error": "No files uploaded"}, 400

    files = request.files.getlist('files')
    video_frames = []
    
    # Processing frames
    for idx, file in enumerate(files):
        img = Image.open(BytesIO(file.read())).convert('RGB')
        img_np = np.array(img)

        # Split image into tiles
        tiles, h, w = split_image_into_tiles(img_np, tile_size, tile_overlap)
        processed_tiles = []

        for (x, y, tile) in tiles:
            processed_tiles.append((x, y, tile))

        # Stitch tiles back together
        output_img = stitch_tiles(processed_tiles, h, w, tile_size, tile_overlap)

        # YOLO Inference on the stitched image
        yolo_results = yolo_model(output_img)

        # Annotate the frame with bounding boxes if detections are made
        if yolo_results and len(yolo_results[0].boxes) > 0:
            for box in yolo_results[0].boxes:
                # Draw bounding boxes on the frame
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{box.cls}: {box.conf:.2f}'
                cv2.putText(output_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert back to 8-bit for video writing
        output_img_uint8 = (output_img * 255.0).clip(0, 255).astype(np.uint8)
        video_frames.append(output_img_uint8)

    # Create video from processed frames
    video_path = create_video(video_frames)

    # Stream the video to the streamer server
    stream_to_server(video_path)

    return {"message": "Video processed and sent successfully"}, 200

def create_video(frames, fps=20, video_name='output_video.avi'):
    # Get frame dimensions
    height, width, layers = frames[0].shape
    size = (width, height)

    # Define the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_path = os.path.join('results', video_name)
    out = cv2.VideoWriter(video_path, fourcc, fps, size)

    # Write each frame to the video
    for frame in frames:
        out.write(frame)

    out.release()
    return video_path

def stream_to_server(video_path):
    # Send video file to the streamer server
    with open(video_path, 'rb') as video_file:
        files = {'video': video_file}
        response = requests.post(streamer_url, files=files)
        if response.status_code == 200:
            print("Video successfully streamed to the server.")
        else:
            print(f"Failed to stream video. Error: {response.text}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)

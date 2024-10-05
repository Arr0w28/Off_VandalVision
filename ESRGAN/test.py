import os.path as osp
import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO  # YOLOv8 import

# Set the YOLOv8 model path
yolo_model_path = 'yolov8m.pt'  # Path to your YOLOv8m model

# Path to the image folder
test_img_folder = 'LR/comic.png'  # Adjust path if necessary

# Load YOLOv8m model
yolo_model = YOLO(yolo_model_path)  # Load YOLOv8 model

# Tiling settings
tile_size = 128  # Adjust this based on your memory limit
tile_overlap = 32  # Overlap to avoid seams between tiles

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

# Process each image
idx = 0
img_paths = glob.glob(test_img_folder)

# Ensure the results directory exists
results_dir = 'results/'
if not osp.exists(results_dir):
    os.makedirs(results_dir)

for path in img_paths:
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(f'Processing image {idx}: {base}')

    # Read image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error reading image {path}")
        continue

    # Split image into tiles
    tiles, h, w = split_image_into_tiles(img, tile_size, tile_overlap)
    processed_tiles = []

    # Process each tile directly (No ESRGAN involved)
    for (x, y, tile) in tiles:
        processed_tiles.append((x, y, tile))

    # Stitch the tiles back together
    output_img = stitch_tiles(processed_tiles, h, w, tile_size, tile_overlap)

    # Save the stitched image
    result_path = osp.join(results_dir, f'{base}_stitched.png')
    cv2.imwrite(result_path, output_img)
    print(f'Saved stitched image to {result_path}')

    # YOLO Inference on the stitched image
    results = yolo_model(result_path)  # Perform YOLO prediction on the stitched image

    # Process YOLO results (e.g., visualize bounding boxes, or output classification results)
    if results and len(results[0].boxes) > 0:  # Check if any boxes were detected
        print(f"Vandalism detected in image {base}")
    else:
        print(f"No vandalism detected in image {base}")
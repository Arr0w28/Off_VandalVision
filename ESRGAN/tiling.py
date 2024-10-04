import os.path as osp
import os
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

# Set the model path
model_path = 'models/RRDB_ESRGAN_x4.pth'  # You can use ESRGAN model or PSNR model here
device = torch.device('mps')  # Ensure that you're using CPU as no CUDA support on M1

# Path to the image folder
test_img_folder = 'LR/Normalised-CCTV-face-images-2.png'  # Adjust path if necessary

# Initialize the model architecture
model = arch.RRDBNet(3, 3, 64, 23, gc=32)

# Load model weights
model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
model.eval()  # Set the model to evaluation mode
model = model.to(device)  # Send the model to the device (CPU)

print(f'Model loaded from {model_path}. \nStarting inference...')

# Glob to match the image file pattern, adjust if needed
img_paths = glob.glob(test_img_folder)

# Ensure the results directory exists
results_dir = 'results/'
if not osp.exists(results_dir):
    os.makedirs(results_dir)

# Define the tile size
tile_size = (128, 128)  # You can change the tile size based on your needs

# Function to tile the image
def tile_image(image, tile_size=(128, 128)):
    img_height, img_width, _ = image.shape
    tiles = []
    for y in range(0, img_height, tile_size[1]):
        for x in range(0, img_width, tile_size[0]):
            tile = image[y:y+tile_size[1], x:x+tile_size[0]]
            tiles.append(tile)
    return tiles

# Function to reassemble tiles back into the image
def assemble_tiles(tiles, img_shape, tile_size=(128, 128)):
    output_img = np.zeros(img_shape, dtype=np.uint8)  # Initialize an empty image
    idx = 0
    h, w, _ = img_shape
    for y in range(0, h, tile_size[1]):
        for x in range(0, w, tile_size[0]):
            tile = tiles[idx]
            tile_h, tile_w, _ = tile.shape

            # If the tile is smaller than the specified tile size, pad it
            if tile_h != tile_size[1] or tile_w != tile_size[0]:
                padded_tile = np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8)
                padded_tile[:tile_h, :tile_w, :] = tile  # Place the smaller tile in the padded array
                tile = padded_tile
            
            # Place the tile in the output image
            output_img[y:y+tile_size[1], x:x+tile_size[0]] = tile
            idx += 1
    return output_img


# Process each image
idx = 0
for path in img_paths:
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(f'Processing image {idx}: {base}')

    # Read the image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error reading image {path}")
        continue

    # Normalize image to [0, 1] range
    img = img.astype(np.float32) / 255.0

    # Tile the image
    tiles = tile_image(img, tile_size=tile_size)
    enhanced_tiles = []

    # Process each tile
    for tile in tiles:
        # Prepare tile for the model (convert to torch tensor and change color channels)
        tile_tensor = torch.from_numpy(np.transpose(tile[:, :, [2, 1, 0]], (2, 0, 1))).float()  # Convert to torch tensor
        tile_tensor = tile_tensor.unsqueeze(0).to(device)  # Add batch dimension and send to device

        # Run inference on the tile
        with torch.no_grad():
            output_tile = model(tile_tensor).squeeze(0).cpu().numpy()

        # Convert back to [0, 255] range and BGR format
        output_tile = np.transpose(output_tile[[2, 1, 0], :, :], (1, 2, 0))  # Convert back to BGR
        output_tile = (output_tile * 255.0).clip(0, 255).astype(np.uint8)  # Clip to valid pixel values

        enhanced_tiles.append(output_tile)  # Save the enhanced tile

    # Reassemble the enhanced tiles into the full image
    enhanced_img = assemble_tiles(enhanced_tiles, img.shape, tile_size=tile_size)

    # Save the enhanced image
    result_path = osp.join(results_dir, f'{base}_enhanced.png')
    cv2.imwrite(result_path, enhanced_img)
    print(f'Saved enhanced image to {result_path}')

print("Tiling and enhancement completed.")

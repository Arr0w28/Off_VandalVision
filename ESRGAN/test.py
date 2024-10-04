import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

# Set the model path
model_path = 'models/RRDB_PSNR_x4.pth'  # You can use ESRGAN model or PSNR model here
device = torch.device('cpu')  # Ensure that you're using CPU as no CUDA support on M1

# Path to the image folder
test_img_folder = 'LR/Normalised-CCTV-face-images-2_rlt.png'  # Adjust path if necessary

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

# Process each image
idx = 0
for path in img_paths:
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(f'Processing image {idx}: {base}')

    # Read image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error reading image {path}")
        continue

    # Normalize image to [0, 1] range
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()  # Change color channels

    # Add batch dimension and move to device
    img_LR = img.unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(img_LR).squeeze(0).cpu().numpy()

    # Convert back to [0, 255] and BGR order for OpenCV
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # Switch RGB to BGR
    output = (output * 255.0).clip(0, 255).astype(np.uint8)  # Clip values to valid range

    # Save output image
    result_path = osp.join(results_dir, f'{base}_rlt.png')
    cv2.imwrite(result_path, output)
    print(f'Saved result to {result_path}')

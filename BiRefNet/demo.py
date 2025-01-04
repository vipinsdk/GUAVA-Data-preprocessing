# Imports
from PIL import Image
import torch
import os
from torchvision import transforms
from .models.birefnet import BiRefNet
from .utils import check_state_dict
from tqdm import tqdm
import argparse

def background_matting(folder_path, output_folder):
    # Load Model
    birefnet = BiRefNet(bb_pretrained=False)
    state_dict = torch.load('./BiRefNet-general-epoch_244.pth', map_location='cpu')
    state_dict = check_state_dict(state_dict)
    birefnet.load_state_dict(state_dict)

    # Load Model
    device = 'cuda'
    torch.set_float32_matmul_precision(['high', 'highest'][0])

    birefnet.to(device)
    birefnet.eval()
    print('BiRefNet is ready to use.')

    # Input Data
    transform_image = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create output directory if it doesn't exist
    images_out = os.path.join(output_folder, 'images')
    masks_out = os.path.join(output_folder, 'fg_masks')
    if not os.path.exists(images_out):
        os.makedirs(images_out, exist_ok=True)
        os.makedirs(masks_out, exist_ok=True)

    # Walk through all subfolders and files
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image_files.append(os.path.join(root, file))

    # Loop through all image files with a progress bar
    for image_file in tqdm(sorted(image_files), desc="Processing images", unit="image"):
        image = Image.open(image_file)
        
        # Process and predict
        input_images = transform_image(image).unsqueeze(0).to('cuda')
        
        with torch.no_grad():
            preds = birefnet(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        
        # Convert predictions to PIL image
        pred_pil = transforms.ToPILImage()(pred)
        
        # Scale proportionally with max length to 1024
        scale_ratio = 1024 / max(image.size)
        scaled_size = (int(image.size[0]), int(image.size[1]))
        # Prepare the image with alpha channel
        image_masked = image.resize((1024, 1024))
        image_masked.putalpha(pred_pil)

        image_file = os.path.splitext(os.path.basename(image_file))[0]
        parts = image_file.split('_')
        parts[0] = str(int(parts[0]) - 1)
        image_file = parts[1][1:] + '_' + parts[0].zfill(2) + '.png'
        
        output_path_mask = os.path.join(images_out, image_file)
        output_path_pred = os.path.join(masks_out, image_file)
        
        image_masked.resize(scaled_size).save(output_path_mask, format="PNG")
        pred_pil.resize(scaled_size).save(output_path_pred, format="PNG")

    print("Processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BiRefNet Image Processing")
    parser.add_argument("--folder_path", type=str, help="Path to the folder containing input images")
    parser.add_argument("--output_folder", type=str, help="Path to the output folder")
    args = parser.parse_args()

    background_matting(args.folder_path, args.output_folder)

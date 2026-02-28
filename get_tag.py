'''
 * SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution 
 * Modified for unified dataset list generation
'''
import torch
from torchvision import transforms
import os
import glob
import argparse
from PIL import Image
from ram.models.ram import ram
from ram import inference_ram as inference
from tqdm import tqdm


ram_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

parser = argparse.ArgumentParser()

parser.add_argument("--ram_path", type=str, default='', help='RAM weight')
parser.add_argument("--hr_path", type=str, default='', help='HR images directory')
parser.add_argument("--lr_path", type=str, default='', help='LR images directory')
parser.add_argument("--out_txt", type=str, default='', help='Path to save the combined metadata txt')
parser.add_argument("--gpu_id", type=int, default=0) 
args = parser.parse_args()


lr_lists = sorted(glob.glob(os.path.join(args.lr_path, '*.png')))
num_imgs = len(lr_lists)

model = ram(pretrained=args.ram_path, image_size=384, vit='swin_l')
model = model.eval().to('cuda')
device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
# Start processing and writing to file
# Using 'w' mode will overwrite the file if it already exists
with torch.no_grad(), open(args.out_txt, "w") as f:
    # Use tqdm for a progress bar with estimated time
    for lq_path in tqdm(lr_lists, desc="Tagging progress"):
        try:
            basename = os.path.basename(lq_path)
            
            # --- Filename matching logic ---
            hr_path = os.path.join(args.hr_path, basename)
            
            # Check if corresponding HR exists
            if not os.path.exists(hr_path):
                # print(f"Warning: HR image not found for: {hr_path}")
                continue

            img_input = ram_transforms(Image.open(lq_path)).unsqueeze(0).to(device)
            captions = inference(img_input, model)
        
            tag_text = captions[0]
            # Absolute paths are used to ensure the training script can locate files from any directory
            lq_abs = os.path.abspath(lq_path)
            hr_abs = os.path.abspath(hr_path)
            f.write(f"{lq_abs} {hr_abs} {tag_text}\n")
            
        except Exception as e:
            print(f"\nError processing {lq_path}: {e}")

print(f"\n>> Task completed! Metadata saved to: {args.out_txt}")
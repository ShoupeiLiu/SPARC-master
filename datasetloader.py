import numpy as np
import os
import tifffile as tiff
from torch.utils.data import Dataset
import random
import torch

def random_transform(input, label):
    p_trans = random.randrange(4)
    if p_trans == 0:
        pass
    elif p_trans == 1:
        input = np.rot90(input, k=2, axes=(1, 2))
        label = np.rot90(label, k=2, axes=(0, 1))
    elif p_trans == 2:
        input = input[:, ::-1, :]
        label = label[::-1, :]
    elif p_trans == 3:
        input = input[:, :, ::-1]
        label = label[:, ::-1]
    return input, label

class InMemoryPatchDataset(Dataset):
    def __init__(self, raw_data_folder, label_data_folder, patch_t=32, patch_y=32, patch_x=128, overlap_factor=0.5, target_patches=1000):
        """
        Parameter Description:
        target_patches: The number of training data pairs you want to generate.
                        If set to 1000, only 1000 pairs are generated, and the rest are ignored.
                        If set to None, all available data is generated.
        """
        self.patch_t = patch_t
        self.patch_y = patch_y
        self.patch_x = patch_x
        self.overlap_factor = overlap_factor
        self.target_patches = target_patches
        
        self.input_patches = []
        self.label_patches = []
        
        self.gap_t = int(patch_t * (1 - overlap_factor))
        self.gap_y = int(patch_y * (1 - overlap_factor))
        self.gap_x = int(patch_x * (1 - overlap_factor))
        print(raw_data_folder)
        raw_files = sorted([f for f in os.listdir(raw_data_folder) if f.endswith(('.tif', '.tiff'))])
        print(f"🎯 Target data quantity: {target_patches} pairs")
        print(f"📂 Scanned {len(raw_files)} files, starting patch generation...")
        
        current_count = 0 # Current number of generated patches
        
        for raw_name in raw_files:
            # If the target quantity is reached, break the loop immediately and stop reading further files
            if self.target_patches is not None and current_count >= self.target_patches:
                print("✅ Target data quantity reached, stopping file reading.")
                break

            im_dir = os.path.join(raw_data_folder, raw_name)
            # Assume label naming convention
            label_name = raw_name.replace('.tif', '') + '_label.tif'
            label_dir = os.path.join(label_data_folder, label_name)
            
            if not os.path.exists(label_dir):
                print(f"⚠️ Warning: Label not found at {label_dir}, skipping this file")
                continue

            noise_im = tiff.imread(im_dir).astype(np.float32)
            noise_label = tiff.imread(label_dir).astype(np.float32)
            
            whole_t, whole_y, whole_x = noise_im.shape
            
            # Patching loop
            for t in range(0, int((whole_t - patch_t + self.gap_t) / self.gap_t)):
                # Check quantity again (check in every loop layer to ensure precision)
                if self.target_patches is not None and current_count >= self.target_patches:
                    break
                    
                for y in range(0, int((whole_y - patch_y + self.gap_y) / self.gap_y)):
                    if self.target_patches is not None and current_count >= self.target_patches:
                        break
                        
                    for x in range(0, int((whole_x - patch_x + self.gap_x) / self.gap_x)):
                        if self.target_patches is not None and current_count >= self.target_patches:
                            break
                            
                        # --- Calculate coordinates and crop patch ---
                        init_s = t * self.gap_t
                        init_h = y * self.gap_y
                        init_w = x * self.gap_x
                        
                        noise_patch1 = noise_im[init_s:init_s+patch_t, init_h:init_h+patch_y, init_w:init_w+patch_x]
                        
                        # Label cropping logic (assuming 4x upsampling)
                        label_h_start = init_h * 4
                        label_h_end = (init_h + patch_y) * 4
                        label_patch1 = noise_label[label_h_start:label_h_end, init_w:init_w+patch_x]
                        
                        # Store in lists
                        self.input_patches.append(noise_patch1)
                        self.label_patches.append(label_patch1)
                        
                        current_count += 1
                        
                        # Check if target is reached immediately after storing
                        if self.target_patches is not None and current_count >= self.target_patches:
                            break
            
            print(f"   - Finished processing {raw_name}, current total samples: {current_count}")

        print(f"🎉 Patch generation complete! Finally generated {len(self.input_patches)} samples.")

    def __len__(self):
        return len(self.input_patches)

    def __getitem__(self, idx):
        input_image = self.input_patches[idx]
        label_image = self.label_patches[idx]
        
        input_image, label_image = random_transform(input_image, label_image)
        
        input_tensor = torch.from_numpy(np.expand_dims(input_image, 0).copy())
        label_tensor = torch.from_numpy(np.expand_dims(label_image, 0).copy())
        
        return input_tensor, label_tensor
import os
import numpy as np
import tifffile as tiff
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasetloader import InMemoryPatchDataset
import torch.optim as optim
import argparse
import time
import datetime
import random
from scipy.linalg import toeplitz
import cv2
from network import UNet3D as UNet3D

def upsample_matrix(frames, reference):
    """
    Custom upsampling function using Toeplitz matrix.
    This is the complex reconstruction algorithm from the original code.
    """
    frames = frames.squeeze(0).squeeze(0) # Remove batch and channel dims -> (T, H, W)
    frames = frames.cpu().numpy().astype(np.float32)
    ref_reshaped = reference.reshape(frames.shape[1], 4, frames.shape[2]) # Reshape reference
    
    # Calculate ratios
    row_sums = ref_reshaped.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    ratios = ref_reshaped / row_sums
    ratios = ratios.reshape(frames.shape[1]*4, frames.shape[2])
    ratios = ratios.cpu().numpy().astype(np.float32)
    
    epsilon = 1e-8
    d = np.arange(1, frames.shape[2] + 1)
    w = 1.0 / (d + epsilon)
    W = toeplitz(w)
    W[W < 0.4] = 0.0
    
    rows = frames.shape[1] * 4
    output = np.zeros([frames.shape[0], reference.shape[0], reference.shape[1]])
    
    for t in range(frames.shape[0]):
        image = frames[t, :, :]
        row_base = np.arange(reference.shape[0]) // 4
        
        X_up_new = np.zeros_like(ratios)
        for i in range(frames.shape[1] * 4):
            row = row_base[i]
            if i < 4:
                idx = np.arange(0, 8)
                img_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1])
            elif i < rows - 4:
                idx_start = 4 * (row - 1) + 0
                idx = np.arange(idx_start, idx_start + 12)
                img_rows = np.array([row-1] * 4 + [row] * 4 + [row+1] * 4)
            else:
                idx = np.arange(rows - 8, rows)
                img_rows = np.array([int(rows/4-2)] * 4 + [int(rows/4-1)] * 4)
                
            weights = W[idx, i]
            img_vals = image[img_rows, :]
            label_vals = ratios[idx, :]
            X_up_new[i, :] = np.sum(weights[:, None] * img_vals * label_vals, axis=0)
            
        output[t, :, :] = X_up_new

    output = torch.from_numpy(output).type(torch.float32)
    return output.unsqueeze(0).unsqueeze(0) # Add batch and channel dims back

def scale_to_one(tensor):
    """Normalize tensor values to range [0, 1] for visualization."""
    t_min, t_max = tensor.min(), tensor.max()
    return (tensor - t_min) / (t_max - t_min + 1e-8)

def main():
    # --- Argument Parser ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument('--GPU', type=str, default='0', help="GPU index")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--patch_y', type=int, default=32, help="Temporal patch size")
    parser.add_argument('--patch_x', type=int, default=128, help="Temporal patch size")
    parser.add_argument('--patch_t', type=int, default=32, help="Temporal patch size")
    parser.add_argument('--data_num', type=int, default=50000, help="Temporal patch size")
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument("--b1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--b2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument('--fmap', type=int, default=16, help='Feature map channels')
    parser.add_argument('--save_freq', type=int, default=5, help='Save model every N epochs')
    
    # --- Paths to your raw data folders ---
    parser.add_argument('--raw_data_folder', type=str, default='datasets/train/Input', help='Folder containing raw .tif files')
    parser.add_argument('--label_data_folder', type=str, default='datasets/train/Reference', help='Folder containing label _label.tif files')
    
    opt = parser.parse_args()
    
    # --- Device Setup ---
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_time = 'Train' + '_' + datetime.datetime.now().strftime("%Y%m%d%H%M")
    pth_path = os.path.join('pth', current_time)
    os.makedirs(pth_path, exist_ok=True)
    # --- Loss Functions ---
    L2_pixelwise = nn.MSELoss().cuda()
    L1_pixelwise = nn.L1Loss().cuda()
    
    # --- Model Setup ---
    SPARC_generator = UNet3D(in_channels=1, out_channels=1, f_maps=opt.fmap)
    SPARC_generator = SPARC_generator.to(device)
    
    # --- Optimizer ---
    optimizer_G = optim.Adam(SPARC_generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=opt.n_epochs)
    
    # --- Create Dataset and Dataloader ---
    # This step will take time as it loads all data into RAM
    try:
        train_dataset = InMemoryPatchDataset(
            raw_data_folder=opt.raw_data_folder,
            label_data_folder=opt.label_data_folder,
            patch_t=opt.patch_t,
            patch_y=opt.patch_y, # Fixed from original code
            patch_x=opt.patch_x, # Fixed from original code
            overlap_factor=0.5,
            target_patches=opt.data_num
        )
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check if the paths are correct and files exist.")
        return

    # --- Training Loop ---
    print("🚀 Starting training...")
    for epoch in range(opt.n_epochs):
        SPARC_generator.train()
        epoch_losses = []
        start_time = time.time()

        for iteration, (noisy, label) in enumerate(train_loader):
            noisy = noisy.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True).squeeze(1) # Remove channel dim for processing
            
            # --- Forward Pass Logic (Matching original code structure) ---
            # Split input into even and odd frames
            noisy_patch1 = noisy[:, :, 0:opt.patch_t:2, :, :] # Even frames (input)
            noisy_patch2 = noisy[:, :, 1:opt.patch_t:2, :, :] # Odd frames (target)
            # Simulate model output (Replace with actual model call in real use)
            # The model should output denoised_B (prediction) and upsample_B (feature)
            # For this demo, we use dummy outputs or pass through
            if hasattr(SPARC_generator, 'dummy'):
                # If using the dummy model, create dummy outputs
                denoised_B = torch.zeros_like(noisy_patch2)
                upsample_B = torch.zeros(noisy_patch2.size(0), 1, noisy_patch2.size(2), 128, noisy_patch2.size(4)) # Guess shape
            else:
                # Real model call
                denoised_B, upsample_B = SPARC_generator(noisy_patch1)

            # --- Target Generation for Upsampling Loss ---
            try:
                target_upsample = upsample_matrix(noisy_patch2, label[0]) # Using the custom matrix function
            except Exception as e:
                print(f"Error in upsample_matrix: {e}")
                target_upsample = torch.zeros_like(upsample_B)
            target_upsample = target_upsample.cuda()
            # --- Loss Calculation ---
            # Using dummy shapes if necessary
            l2_loss = L2_pixelwise(denoised_B, noisy_patch2).cuda()
            l1_loss = L1_pixelwise(denoised_B, noisy_patch2).cuda()
            # Note: The upsample loss is commented out here to avoid dim mismatch in the demo
            l2_loss += L2_pixelwise(upsample_B, target_upsample)
            l1_loss += L1_pixelwise(upsample_B, target_upsample)
            
            total_loss = 0.5 * (l2_loss + l1_loss)

            # --- Backward Pass ---
            optimizer_G.zero_grad()
            total_loss.backward()
            optimizer_G.step()

            epoch_losses.append(total_loss.item())

            # --- Logging ---
            if (iteration + 1) % 10 == 0: # Change to 50 for large datasets
                elapsed = time.time() - start_time
                steps_done = iteration + 1
                eta_seconds = (elapsed / steps_done) * (len(train_loader) - steps_done)
                print(f"[Epoch {epoch}/{opt.n_epochs}] [Batch {iteration}/{len(train_loader)}] [Loss: {total_loss.item():.4f}] [ETA: {datetime.timedelta(seconds=int(eta_seconds))}]")

        # --- Epoch End: Save Model ---
        scheduler_G.step()
        if (epoch) % opt.save_freq == 0:
            # Create save directory
            save_dir = pth_path
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
            torch.save(SPARC_generator.state_dict(), model_path)
            print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()


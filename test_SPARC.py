import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import time
import datetime
import numpy as np
from network import UNet3D as UNet3D
from torch.autograd import Variable
from tifffile import TiffWriter
from test_data_process import test_preprocess_lessMemoryNoTail_chooseOne, testset, singlebatch_test_save, singlebatch_test_save_hs
#%%
# Argument Parser
parser = argparse.ArgumentParser()

parser.add_argument('--GPU', type=str, default='0', help="GPU index for computation (e.g., '0', '0,1')")
parser.add_argument('--patch_x', type=int, default=128, help="Patch width (x-axis)")
parser.add_argument('--patch_y', type=int, default=32, help="Patch height (y-axis)")
parser.add_argument('--patch_t', type=int, default=32, help="Patch depth (t-axis)")
parser.add_argument('--overlap_factor', type=float, default=0.5, help="Overlap factor between adjacent patches")
parser.add_argument('--datasets_path', type=str, default='datasets', help="Root path of datasets")
parser.add_argument('--datasets_folder', type=str, default='test', help="Folder containing files for inference")
parser.add_argument('--pth_path', type=str, default='./pth', help="Path to model .pth files")
parser.add_argument('--key_word', type=str, default='', help="Keyword filter for models")
parser.add_argument('--model', type=str, default='model', help='Folder containing models to be tested')
parser.add_argument('--fmap', type=int, default=16, help='Number of feature maps in U-Net')
parser.add_argument('--test_datasize', type=int, default=10000000, help='datasets size for test (how many frames)')
parser.add_argument('--scale_factor', type=int, default=1, help='the factor for image intensity scaling')
parser.add_argument('--output_dir', type=str, default='./results', help="Output directory for results")
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU

#%%
# Model Initialization


model_path = os.path.join(opt.pth_path, opt.model)
print("Model path:", model_path)

# Get list of model files
model_list = list(os.walk(model_path, topdown=False))[-1][-1]
model_list.sort()

# Calculate patch gaps based on overlap
opt.gap_x = int(opt.patch_x * (1 - opt.overlap_factor))
opt.gap_y = int(opt.patch_y * (1 - opt.overlap_factor))
opt.gap_t = int(opt.patch_t * (1 - opt.overlap_factor))

# Set batch size equal to number of GPUs
opt.ngpu = opt.GPU.count(',') + 1
opt.batch_size = opt.ngpu

print('\033[1;31mInference parameters: \033[0m')
print(opt)

# Initialize Generator (UNet3D)
SPARC_generator = UNet3D(in_channels=1, out_channels=1, f_maps=opt.fmap)

if torch.cuda.is_available():
    SPARC_generator = SPARC_generator.cuda()
    SPARC_generator = nn.DataParallel(SPARC_generator, device_ids=range(opt.ngpu))
    print('\033[1;31mUsing {} GPU(s) for inference -----> \033[0m'.format(torch.cuda.device_count()))

# Print parameter count
param_num = sum([param.nelement() for param in SPARC_generator.parameters()])
print('\033[1;31mParameters of the model: {:.2f} M. \033[0m'.format(param_num / 1e6))

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Create output directory
current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
output_path1 = os.path.join(opt.output_dir, 'DataFolderIs_' + opt.datasets_folder + '_' + current_time + '_ModelFolderIs_' + opt.model)
if not os.path.exists(output_path1):
    os.mkdir(output_path1)

# Get image list for processing
im_folder = os.path.join(opt.datasets_path, opt.datasets_folder)
img_list = list(os.walk(im_folder, topdown=False))[-1][-1]
img_list.sort()

def test():
    """Main inference function."""
    # Process each model in the model list
    for pth_index in range(len(model_list)):
        if '.pth' not in model_list[pth_index]:
            continue
            
        pth_name = model_list[pth_index]
        print(f"\nLoading model: {pth_name}")
        
        # Create output subdirectory for this model
        model_output_path = os.path.join(output_path1, pth_name.replace('.pth', ''))
        if not os.path.exists(model_output_path):
            os.mkdir(model_output_path)

        # Load model weights
        model_name = os.path.join(opt.pth_path, opt.model, pth_name)
        if isinstance(SPARC_generator, nn.DataParallel):
            SPARC_generator.module.load_state_dict(torch.load(model_name))
        else:
            SPARC_generator.load_state_dict(torch.load(model_name))
        
        SPARC_generator.eval() # Set to evaluation mode
        SPARC_generator.cuda()

        # Process each image stack
        for N in range(len(img_list)):
            # Preprocess: Load image and generate coordinates for patching
            name_list, noise_img, coordinate_list = test_preprocess_lessMemoryNoTail_chooseOne(opt, N)
            
            # Record start time
            time_start = time.time()
            T, H, W = noise_img.shape

            # Use memmap to handle large data without exhausting RAM
            denoise_memmap_path = os.path.join(model_output_path, f'denoise_temp_{N}.dat')
            upsample_memmap_path = os.path.join(model_output_path, f'upsample_temp_{N}.dat')
            
            denoise_img = np.memmap(denoise_memmap_path, dtype='float32', mode='w+', shape=(T, H, W))
            upsampled_img = np.memmap(upsample_memmap_path, dtype='float32', mode='w+', shape=(T, H * 4, W))

            # Prepare result filename
            result_name = os.path.join(model_output_path, f"{img_list[N].replace('.tif', '')}_{pth_name.replace('.pth', '')}_output")
            print(f"Processing: {img_list[N]} -> {result_name}")

            # Create DataLoader for patches
            test_data = testset(name_list, coordinate_list, noise_img)
            testloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)

            # Inference Loop
            with torch.no_grad():
                for iteration, (noise_patch, single_coordinate) in enumerate(testloader):
                    noise_patch = noise_patch.cuda()
                    real_A = Variable(noise_patch)

                    # Forward pass
                    fake_B, upsample_B = SPARC_generator(real_A)
                    
                    # Convert to numpy
                    output_image_1 = np.squeeze(fake_B.cpu().numpy())
                    output_image_2 = np.squeeze(upsample_B.cpu().numpy())

                    # Calculate ETA
                    batches_done = iteration
                    batches_left = len(testloader) - batches_done
                    time_left_seconds = int(batches_left * (time.time() - time.time())) # Note: This logic is flawed, usually uses prev_time
                    time_cost = time.time() - time_start

                    # Print progress
                    print(
                        '\r[Model %d/%d] [Stack %d/%d, %s] [Patch %d/%d] [Time: %.0fs] [ETA: %ss]'
                        % (pth_index + 1, len(model_list), N + 1, len(img_list), img_list[N],
                           iteration + 1, len(testloader), time_cost, time_left_seconds),
                        end=' ')

                    # Save the patch into the corresponding position
                    if (iteration + 1) % 1 == 0: # Always true, can be removed
                        aaaa, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = singlebatch_test_save(single_coordinate, output_image_1)
                        denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] = aaaa
                        
                        aaaa, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = singlebatch_test_save_hs(single_coordinate, output_image_2)
                        upsampled_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] = aaaa

                # --- End of DataLoader Loop ---

            # Save final results as TIF
            result_name1 = result_name + '_denoise.tif'
            result_name2 = result_name + '_upsample.tif'
            
            with TiffWriter(result_name1, bigtiff=True) as writer:
                for t in range(T):
                    slice_t = (denoise_img[t] * opt.scale_factor).astype('int16')
                    writer.write(slice_t, contiguous=True)
                    
            with TiffWriter(result_name2, bigtiff=True) as writer:
                for t in range(T):
                    slice_t = (upsampled_img[t] * opt.scale_factor).astype('int16')
                    writer.write(slice_t, contiguous=True)

            print(f"\nResult saved: {result_name1} and {result_name2}")
            
            # Cleanup temporary memmap files
            del denoise_img, upsampled_img
            try:
                os.remove(denoise_memmap_path)
                os.remove(upsample_memmap_path)
            except Exception as e:
                print(f"Warning: Could not delete temp files: {e}")

if __name__ == "__main__":
    test()
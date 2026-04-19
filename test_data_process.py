import numpy as np
import os
import tifffile as tiff
from skimage import io
import random
import math
import torch
from torch.utils.data import Dataset
from skimage import io
#%%
class testset(Dataset):
    def __init__(self,name_list,coordinate_list,noise_img):
        self.name_list = name_list
        self.coordinate_list=coordinate_list
        self.noise_img = noise_img

    def __getitem__(self, index):
        #fn = self.images[index]
        single_coordinate = self.coordinate_list[self.name_list[index]]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']
        noise_patch = self.noise_img[init_s:end_s, init_h:end_h, init_w:end_w]
        noise_patch = torch.from_numpy(np.expand_dims(noise_patch, 0))
        #target = self.target[index]
        return noise_patch, single_coordinate
    def __len__(self):
        return len(self.name_list)
def test_preprocess_lessMemoryNoTail_chooseOne(args, N):
    patch_y = args.patch_y
    patch_x = args.patch_x
    patch_t2 = args.patch_t
    gap_y = args.gap_y
    # gap_y = 14
    gap_x = args.gap_x
    gap_t2 = args.gap_t
    cut_w = (patch_x - gap_x)/2
    cut_h = (patch_y - gap_y)/2
    cut_s = (patch_t2 - gap_t2)/2
    # print(gap_y)
    # print(patch_y)
    # print('-=----')
    # print(cut_w)
    # print(cut_h)
    # print(cut_s)
    # print('aaaaa')
    # assert cut_w >= 0 and cut_h >= 0 and cut_s >= 0, "test cut size is negative!"
    im_folder = args.datasets_path + '//' + args.datasets_folder
    name_list = []
    # train_raw = []
    coordinate_list = {}
    img_list = list(os.walk(im_folder, topdown=False))[-1][-1]
    img_list.sort()
    # print(img_list)
    im_name = img_list[N]
    im_dir = im_folder + '//' + im_name
    noise_im = tiff.imread(im_dir)

    if noise_im.shape[0] > args.test_datasize:
        noise_im = noise_im[0:args.test_datasize, :, :]
    #noise_im = noise_im + 1
    noise_im = noise_im.astype(np.float32) / args.scale_factor
    #print(noise_im.mean())
    # noise_im = noise_im - 32767
    # noise_im = noise_im-noise_im.mean()

    whole_x = noise_im.shape[2]
    whole_y = noise_im.shape[1]
    whole_t = noise_im.shape[0]

    num_w = math.ceil((whole_x - patch_x + gap_x) / gap_x)
    num_h = math.ceil((whole_y - patch_y + gap_y) / gap_y)
    num_s = math.ceil((whole_t - patch_t2 + gap_t2) / gap_t2)
    # print(num_w)
    # print(num_h)
    # print(num_s)
    for z in range(0, num_s):
        for x in range(0, num_h):
            for y in range(0, num_w):
                single_coordinate = {'init_h': 0, 'end_h': 0, 'init_w': 0, 'end_w': 0, 'init_s': 0, 'end_s': 0}
                if x != (num_h - 1):
                    init_h = gap_y * x
                    end_h = gap_y * x + patch_y
                elif x == (num_h - 1):
                    init_h = whole_y - patch_y
                    end_h = whole_y
                if y != (num_w - 1):
                    init_w = gap_x * y
                    end_w = gap_x * y + patch_x
                elif y == (num_w - 1):
                    init_w = whole_x - patch_x
                    end_w = whole_x
                if z != (num_s - 1):
                    init_s = gap_t2 * z
                    end_s = gap_t2 * z + patch_t2
                elif z == (num_s - 1):
                    init_s = whole_t - patch_t2
                    end_s = whole_t
                single_coordinate['init_h'] = init_h
                single_coordinate['end_h'] = end_h
                single_coordinate['init_w'] = init_w
                single_coordinate['end_w'] = end_w
                single_coordinate['init_s'] = init_s
                single_coordinate['end_s'] = end_s


                if y == 0:
                    single_coordinate['stack_start_w'] = y*gap_x
                    single_coordinate['stack_end_w'] = y*gap_x+patch_x-cut_w
                    single_coordinate['patch_start_w'] = 0
                    single_coordinate['patch_end_w'] = patch_x-cut_w


                elif y == num_w-1:
                    single_coordinate['stack_start_w'] = whole_x-patch_x+cut_w
                    single_coordinate['stack_end_w'] = whole_x
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = patch_x

                else:
                    single_coordinate['stack_start_w'] = y*gap_x+cut_w
                    single_coordinate['stack_end_w'] = y*gap_x+patch_x-cut_w
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = patch_x-cut_w

#####################################################################################
                if x == 0:

                    # single_coordinate['stack_start_h'] = x*gap_y
                    # single_coordinate['stack_end_h'] = x*gap_y+patch_y-cut_h
                    # single_coordinate['patch_start_h'] = 0
                    # single_coordinate['patch_end_h'] = patch_y-cut_h
                    if num_h == 1:

                        single_coordinate['stack_start_h'] = x*gap_y
                        single_coordinate['stack_end_h'] = patch_y
                        single_coordinate['patch_start_h'] = 0
                        single_coordinate['patch_end_h'] = patch_y
                        # print('1111')
                        # print(single_coordinate['stack_start_h'])
                        # print(single_coordinate['stack_end_h'])
                        # print(single_coordinate['patch_start_h'])
                        # print(single_coordinate['patch_end_h'])
                    else:
                        # print('222')
                        single_coordinate['stack_start_h'] = x*gap_y
                        single_coordinate['stack_end_h'] = x*gap_y+patch_y-cut_h
                        single_coordinate['patch_start_h'] = 0
                        single_coordinate['patch_end_h'] = patch_y-cut_h

                elif x == num_h-1:
                    single_coordinate['stack_start_h'] = whole_y-patch_y+cut_h
                    single_coordinate['stack_end_h'] = whole_y
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = patch_y
                    # print('2222')
                    # print(single_coordinate['stack_start_h'])
                    # print(single_coordinate['stack_end_h'])
                    # print(single_coordinate['patch_start_h'])
                    # print(single_coordinate['patch_end_h'])
                else:
                    single_coordinate['stack_start_h'] = x*gap_y+cut_h
                    single_coordinate['stack_end_h'] = x*gap_y+patch_y-cut_h
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = patch_y-cut_h
######################################################################################
                if z == 0:
                    single_coordinate['stack_start_s'] = z * gap_t2
                    single_coordinate['stack_end_s'] = z * gap_t2 + patch_t2 - cut_s
                    single_coordinate['patch_start_s'] = 0
                    single_coordinate['patch_end_s'] = patch_t2 - cut_s
                elif z == num_s - 1:
                    single_coordinate['stack_start_s'] = whole_t - patch_t2 + cut_s
                    single_coordinate['stack_end_s'] = whole_t
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = patch_t2
                else:
                    single_coordinate['stack_start_s'] = z * gap_t2 + cut_s
                    single_coordinate['stack_end_s'] = z * gap_t2 + patch_t2 - cut_s
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = patch_t2 - cut_s

                # noise_patch1 = noise_im[init_s:end_s,init_h:end_h,init_w:end_w]
                patch_name = args.datasets_folder + '_x' + str(x) + '_y' + str(y) + '_z' + str(z)
                # train_raw.append(noise_patch1.transpose(1,2,0))
                name_list.append(patch_name)
                # print(' single_coordinate -----> ',single_coordinate)
                coordinate_list[patch_name] = single_coordinate

    return name_list, noise_im, coordinate_list

def singlebatch_test_save(single_coordinate, output_image):
    stack_start_w = int(single_coordinate['stack_start_w'])
    stack_end_w = int(single_coordinate['stack_end_w'])
    patch_start_w = int(single_coordinate['patch_start_w'])
    patch_end_w = int(single_coordinate['patch_end_w'])

    stack_start_h = int(single_coordinate['stack_start_h'])
    stack_end_h = int(single_coordinate['stack_end_h'])
    patch_start_h = int(single_coordinate['patch_start_h'])
    patch_end_h = int(single_coordinate['patch_end_h'])

    stack_start_s = int(single_coordinate['stack_start_s'])
    stack_end_s = int(single_coordinate['stack_end_s'])
    patch_start_s = int(single_coordinate['patch_start_s'])
    patch_end_s = int(single_coordinate['patch_end_s'])
    aaaa = output_image[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]
    return aaaa,  stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s

def singlebatch_test_save_hs(single_coordinate, output_image):
    stack_start_w = int(single_coordinate['stack_start_w'])
    stack_end_w = int(single_coordinate['stack_end_w'])
    patch_start_w = int(single_coordinate['patch_start_w'])
    patch_end_w = int(single_coordinate['patch_end_w'])

    stack_start_h = int(single_coordinate['stack_start_h'])
    stack_end_h = int(single_coordinate['stack_end_h'])
    patch_start_h = int(single_coordinate['patch_start_h'])
    patch_end_h = int(single_coordinate['patch_end_h'])
    stack_start_h = stack_start_h*4
    stack_end_h = stack_end_h*4
    patch_start_h = patch_start_h*4
    patch_end_h = patch_end_h*4



    stack_start_s = int(single_coordinate['stack_start_s'])
    stack_end_s = int(single_coordinate['stack_end_s'])
    patch_start_s = int(single_coordinate['patch_start_s'])
    patch_end_s = int(single_coordinate['patch_end_s'])
    aaaa = output_image[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]
    # bbbb = raw_image[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]
    return aaaa,  stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s
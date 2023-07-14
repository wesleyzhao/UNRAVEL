#!/usr/bin/env python3

import math, time
import numpy as np
from aicspylibczi import CziFile # source code https://allencellmodeling.github.io/aicspylibczi/_modules/aicspylibczi/CziFile.html
import nibabel as nib
import argparse, os

# downsampling test libs
from skimage.transform import rescale, downscale_local_mean
import dask.array as da

from memory_profiler import profile

"""
 Example Usage for Benchmarking memory
 python3 -m memory_profiler lazy_load_czi.py --input_file /Users/wesley/projects/heifets-wesley-folder/sampleczi.czi --downsample --output_folder /Users/wesley/projects/heifets-wesley-folder --downscale_factor 10

 Example Usages

 _ downsample _
 python3 lazy_load_czi.py --input_file /path/to/your/input.czi --downsample --output_folder /path/to/your/output_folder --downscale_factor 5 --methods rescale local_mean voxel_drop dask

 _ convert without downsampling _
 python3 lazy_load_czi.py --input_file /path/to/your/input.czi --output_file /path/to/your/output.nii.gz

 _ works on wesleys mac _
 python3 lazy_load_czi.py --input_file /Users/wesley/projects/heifets-wesley-folder/sampleczi.czi --output_file /Users/wesley/projects/heifets-wesley-folder/sampleczi-test.czi --downsample --output_folder /Users/wesley/projects/heifets-wesley-folder --downscale_factor 5

"""

@profile
def downsample_skimage_rescale(image, downscale_factor):
    """ Downsample using skimage rescale (applies a Gaussian filter before downsampling to reduce aliasing artifacts) """
    return rescale(image, 1 / downscale_factor, preserve_range=True, anti_aliasing=True)

@profile
def downsample_skimage_local_mean(image, downscale_factor):
    """
    downscale_local_mean computes mean of local blocks of the input image
    and uses these mean values to create a downsampled image.
    This helps preserve local intensity information, but may introduce blockiness
    """
    downscale_factor = int(downscale_factor)
    return downscale_local_mean(image, (downscale_factor, downscale_factor, downscale_factor))

@profile
def downsample_voxel_drop(image, downscale_factor):
    downscale_factor = int(downscale_factor)
    return image[::downscale_factor, ::downscale_factor, ::downscale_factor]

@profile
def downsample_voxel_drop_middle(image, downscale_factor):
    downscale_factor = int(downscale_factor)
    start_index = downscale_factor // 2 if downscale_factor % 2 == 1 else (downscale_factor - 1) // 2
    return image[start_index::downscale_factor, start_index::downscale_factor, start_index::downscale_factor]

@profile
def downsample_voxel_drop_dask(image, downscale_factor):
    downscale_factor = int(downscale_factor)
    image_da = da.from_array(image, chunks=(100, 100, 100))  # chunk size will depend on your memory size
    return image_da[::downscale_factor, ::downscale_factor, ::downscale_factor].compute()

@profile
def downsample_voxel_drop_by_slice(czi, downscale_factor):
    downscale_factor = int(downscale_factor)
    size = czi.size
    num_z, num_y, num_x = size[-3:]
    downsampled_slices = []

    for z in range(0, num_z, downscale_factor):
        img_slice, _ = czi.read_image(C=0, Z=z)
        img_slice = np.squeeze(img_slice)
        downsampled_slices.append(img_slice[::downscale_factor, ::downscale_factor])

    img_downsampled = np.stack(downsampled_slices, axis=0)
    img_downsampled = np.transpose(img_downsampled, (2, 1, 0))

    return img_downsampled

@profile
def downsample_voxel_drop_by_slice_mem(czi, downscale_factor):
    downscale_factor = int(downscale_factor)
    size = czi.size
    num_z, num_y, num_x = size[-3:]

    # calculate the size of the output array
    down_num_z = math.ceil(num_z / downscale_factor)
    down_num_y = math.ceil(num_y / downscale_factor)
    down_num_x = math.ceil(num_x / downscale_factor)

    # preallocate the output array
    img_downsampled = np.empty((down_num_z, down_num_y, down_num_x), dtype=np.uint16)  # assuming 16-bit integers

    for z in range(0, num_z, downscale_factor):
        img_slice, _ = czi.read_image(C=0, Z=z)
        img_slice = np.squeeze(img_slice)
        img_slice_down = img_slice[::downscale_factor, ::downscale_factor]

        # Calculate the effective end coordinates after downsampling
        down_end_y = min((img_slice.shape[0] // downscale_factor) * downscale_factor, down_num_y)
        down_end_x = min((img_slice.shape[1] // downscale_factor) * downscale_factor, down_num_x)

        img_downsampled[z//downscale_factor, :down_end_y, :down_end_x] = img_slice_down[:down_end_y, :down_end_x]

    img_downsampled = np.transpose(img_downsampled, (2, 1, 0))

    return img_downsampled

@profile
def downsample_voxel_drop_by_slice_mem_dask(czi, downscale_factor):
    downscale_factor = int(downscale_factor)
    size = czi.size
    num_z, num_y, num_x = size[-3:]

    # calculate the size of the output array
    down_num_z = math.ceil(num_z / downscale_factor)
    down_num_y = math.ceil(num_y / downscale_factor)
    down_num_x = math.ceil(num_x / downscale_factor)

    # preallocate the output array
    img_downsampled = da.empty((down_num_z, down_num_y, down_num_x), dtype=np.uint16, chunks=(100, 100, 100))  # chunk size will depend on your memory size

    for z in range(0, num_z, downscale_factor):
        img_slice, _ = czi.read_image(C=0, Z=z)
        img_slice = np.squeeze(img_slice)
        img_slice_down = img_slice[::downscale_factor, ::downscale_factor]

        # Calculate the effective end coordinates after downsampling
        down_end_y = min((img_slice.shape[0] // downscale_factor) * downscale_factor, down_num_y)
        down_end_x = min((img_slice.shape[1] // downscale_factor) * downscale_factor, down_num_x)

        img_downsampled[z//downscale_factor, :down_end_y, :down_end_x] = img_slice_down[:down_end_y, :down_end_x]

    img_downsampled = da.transpose(img_downsampled, (2, 1, 0))

    return img_downsampled.compute()

@profile
def downsample_dask(image, downscale_factors):
    """
    dask coarsen function applies a reduction function (eg, mean) to equally-sized blocks of image.
    The size of the blocks is defined by the downscale_factors.
    This method allows for efficient, parallelized downsampling of large images.
    """
    dask_data = da.from_array(image)
    downsampled_dask_data = da.coarsen(da.mean, dask_data, {i: int(round(f)) for i, f in enumerate(downscale_factors)}, trim_excess=True)
    return downsampled_dask_data.compute()

# default input/output file paths
input_file = "/home/bear/Documents/Wesley/czi_test/input.czi"
output_file = "/home/bear/Documents/Wesley/czi_test/output.nii.gz"
output_folder = "/home/bear/Documents/Wesley/czi_test"
downscale_factor = 10

def parse_args():
    """ Parse command line arguments """

    # default input/output file paths
    input_file = "/home/bear/Documents/Wesley/czi_test/input.czi"
    output_file = "/home/bear/Documents/Wesley/czi_test/output.nii.gz"
    output_folder = "/home/bear/Documents/Wesley/czi_test"
    downscale_factor = 10
    default_methods = ['rescale',
                        'local_mean',
                        'voxel_drop',
                        'voxel_drop_middle',
                        'voxel_drop_dask',
                        'drop_by_slice',
                        'drop_by_slice_mem',
                        'drop_by_slice_mem_dask',
                        'dask',
                        ]

    parser = argparse.ArgumentParser(description='Load CZI, downsample, save as NIfTI')
    parser.add_argument('--input_file', default=input_file, help='Path to the input CZI file')
    parser.add_argument('--output_file', default=output_file, help='Path to the output NIfTI file')
    parser.add_argument('--downsample', action='store_true', help='Enable downsampling')
    parser.add_argument('--downscale_factor', type=float, default=downscale_factor, help='Optional downscale factor for downsampling')
    parser.add_argument('--output_folder', default=output_folder, help='Folder to save downsampled images')
    parser.add_argument('--methods', nargs='+', default=default_methods, help='Downsampling methods to use. Options: rescale, local_mean, voxel_drop, dask')
    return parser.parse_args()

@profile
def main():

    # Parse command line arguments
    args = parse_args()

    # Check if output folder exists, if not create it
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    czi = CziFile(args.input_file)

    # Load the first channel of image
    img, shp = czi.read_image(C=0)
    img = np.squeeze(img)
    img = np.transpose(img,(2,1,0))  # transpose image from ZYX to XYZ

    if args.downsample:
        execution_times = {}
        for method in args.methods:
            start_time = time.time()
            if method == 'rescale':
                img_downsampled = downsample_skimage_rescale(img, args.downscale_factor)
            elif method == 'local_mean':
                img_downsampled = downsample_skimage_local_mean(img, args.downscale_factor)
            elif method == 'voxel_drop':
                img_downsampled = downsample_voxel_drop(img, args.downscale_factor)
            elif method == 'voxel_drop_middle':
                img_downsampled = downsample_voxel_drop_middle(img, args.downscale_factor)
            elif method == 'voxel_drop_dask':
                img_downsampled = downsample_voxel_drop_dask(img, args.downscale_factor)
            elif method == 'drop_by_slice':
                img_downsampled = downsample_voxel_drop_by_slice(czi, args.downscale_factor)
            elif method =='drop_by_slice_mem':
                img_downsampled = downsample_voxel_drop_by_slice_mem(czi, args.downscale_factor)
            elif method =='drop_by_slice_mem_dask':
                img_downsampled = downsample_voxel_drop_by_slice_mem_dask(czi, args.downscale_factor)
            elif method == 'dask':
                img_downsampled = downsample_dask(img, (args.downscale_factor, args.downscale_factor, args.downscale_factor))
            else:
                print(f'Invalid method: {method}')
                continue
            nib.save(nib.Nifti1Image(img_downsampled, np.eye(4)), os.path.join(args.output_folder, f"downsample-test-{method}.nii.gz"))
            end_time = time.time()
            execution_time = end_time - start_time
            execution_times[method] = execution_time
        for method, execution_time in execution_times.items():
            print(f"Execution time for {method}: {execution_time} seconds")
    else:
        img = nib.Nifti1Image(img, np.eye(4))
        nib.save(img, args.output_file)

if __name__ == '__main__':
    main()

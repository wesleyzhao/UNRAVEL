#!/usr/bin/env python3

import argparse, os, sys
import nibabel as nib
import numpy as np
import dask.array as da
from aicspylibczi import CziFile
from datetime import datetime
from glob import glob
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Load channel of CZI, [downsample], save as NIfTI', \
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)                               
    czi_files = glob("*.czi")
    default_input = os.path.join(os.getcwd(), czi_files[0]) if czi_files else None
    parser.add_argument('-i', '--input', help='<./img.czi>', default=default_input, metavar='')
    parser.add_argument('-d', '--ds_factor', type=int, default=1, help='Optional factor for downsampling', metavar='')
    parser.add_argument('-m', '--ds_method', default='voxel_drop', help='Other options: dask', metavar='')
    parser.add_argument('-c', '--channel', type=int, default=0, help='0 = 1st channel, 1 = 2nd channel', metavar='')    
    parser.add_argument('-o', '--output', help='<./img.nii.gz> (default: ./input_img[_ds*x].nii.gz>)', metavar='')
    return parser.parse_args()

def downsample_voxel_drop(image, ds_factor):
    ds_factor = int(ds_factor)
    return image[::ds_factor, ::ds_factor, ::ds_factor]

def downsample_dask(image, ds_factor):
    """
    This applies a reduction function (eg, np.mean) to equal-sized blocks of image.
    Block size is defined by ds_factor.
    This method allows for efficient, parallelized downsampling of large images.
    """
    dask_data = da.from_array(image)
    downsampled_dask_data = da.coarsen(da.median, dask_data, {i: int(round(f)) for i, f in enumerate(ds_factor)}, trim_excess=True)
    return downsampled_dask_data.compute()

def main():
    args = parse_args() # Parse command line arguments
    print(str(f"\nRunning {sys.argv[:]} "+datetime.now().strftime("%H:%M:%S")+"\n"))
    print(f"  Arguments: {vars(args)} \n")

    # Check if output folder exists, if not create it
    path = Path(args.input)
    output_folder = path.parent
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    czi = CziFile(args.input)
    print(str(f"  Loading the channel {args.channel} of image "+datetime.now().strftime("%H:%M:%S")))
    img = czi.read_image(C=args.channel)[0] # Load the first channel of image | get img ([1] is the shape)
    print(str(f"  Loaded the channel {args.channel} of image "+datetime.now().strftime("%H:%M:%S")))
    img = np.squeeze(img)    
    img = np.transpose(img,(2,1,0))  # transpose image from ZYX to XYZ

    if args.output is not None:
        output = args.output
    elif args.ds_factor != 1:
        output = path.with_name(path.stem.split('.')[0] + f'_ds{args.ds_factor}x.nii.gz')
    else:
        output = path.with_name(path.stem.split('.')[0] + '.nii.gz')

    if args.ds_factor != 1:
        print(str(f"  Running downsampling method: {args.ds_method} "+datetime.now().strftime("%H:%M:%S")))
        if args.ds_method == 'voxel_drop':
            img_downsampled = downsample_voxel_drop(img, int(args.ds_factor))
            print(str(f"  Finished running: voxel_drop "+datetime.now().strftime("%H:%M:%S")))
        elif args.ds_method == 'dask':
            img_downsampled = downsample_dask(img, (args.ds_factor, args.ds_factor, args.ds_factor))
            print(str(f"  Finished running: dask "+datetime.now().strftime("%H:%M:%S")))
        else:
            print(f'Invalid method: {args.ds_method}')
        print(str(f"  Saving image "+datetime.now().strftime("%H:%M:%S")))
        nib.save(nib.Nifti1Image(img_downsampled, np.eye(4)), output)
        print(str(f"  Finished saving "+datetime.now().strftime("%H:%M:%S"))+"\n")
    else:
        img = nib.Nifti1Image(img, np.eye(4))
        print(str(f"  Saving image "+datetime.now().strftime("%H:%M:%S")))
        nib.save(img, output)
        print(str(f"  Finished saving "+datetime.now().strftime("%H:%M:%S"))+"\n")

if __name__ == '__main__':
    main()
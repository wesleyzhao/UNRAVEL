#!/bin/bash
# metadata.sh
#
# This script generates a metadata text file containing image dimensions
# and voxel size information for a given sample folder.
#
# Usage:
#   ./metadata.sh [--fiji-path /path/to/fiji-binary]
#
# Arguments:
#   --fiji-path: (Optional) Specify the path to the Fiji/ImageJ binary. If not provided,
#                a default path will be used (the variable FIJI_PATH below).
#
# Outputs:
#   A metadata file is created at ./sample??/parameters/metadata
#
# Notes:
#   Alternatively, you can open the image stack or the first image in FIJI
#   and press control+i to view metadata.
#
# (c) Daniel Ryskamp Rijsketic, Boris Heifets @ Stanford University, 2021-2023

if [ "$1" == "help" ]; then
  echo '
Run metadata.sh from the sample?? folder to generate a text file with metadata for getting image dim and voxel size

Outputs:6 ./sample??/parameters/metadata

Alternatively, open stack or first image in FIJI and control+i to view metadata
'
  exit 1
elif [ "$1" == "--fiji-path" ]; then
  IMAGEJ_PATH="$2"
  shift 2
fi

# for cb production
FIJI_PATH="/usr/local/miracl/depends/Fiji.app/ImageJ-linux64"
# for my local env
# FIJI_PATH="/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx"

echo " " ; echo "Running metadata for ${PWD##*/}" ; echo " "

mkdir -p parameters
if [ ! -f parameters/metadata ]; then
  if ls *.czi 1> /dev/null 2>&1; then
    $FIJI_PATH --ij2 -macro metadata $PWD/$(ls *.czi) #Zeiss file type
    mv metadata parameters/metadata
  elif [ -d 488_original ]; then
    cd 488_original
    first_tif=$(ls *.tif | head -1)
    shopt -s nullglob ; for f in *\ *; do mv "$f" "${f// /_}"; done ; shopt -u nullglob #remove spaces from tif series
    $FIJI_PATH --ij2 -macro metadata $PWD/$first_tif
    cd ..
    mv 488_original/metadata parameters/metadata
  else
    cd 488
    first_tif=$(ls *.tif | head -1)
    shopt -s nullglob ; for f in *\ *; do mv "$f" "${f// /_}"; done ; shopt -u nullglob
    $FIJI_PATH --ij2 -macro metadata $PWD/$first_tif
    cd ..
    mv 488/metadata parameters/metadata
  fi
fi


#Daniel Ryskamp Rijsketic 07/07/22 & 07/18/22 & 07/28/22 (Heifets Lab)

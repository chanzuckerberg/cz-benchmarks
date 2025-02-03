#!/bin/bash

# Exit on error
set -e

echo "Copying HVG files..."
cp ../../../../docker/scvi/hvg_names_homo_sapiens.csv.gz ./hvg_names_homo_sapiens.csv.gz
cp ../../../../docker/scvi/hvg_names_mus_musculus.csv.gz ./hvg_names_mus_musculus.csv.gz

echo "Downloading test data from S3..."
aws --profile virtual-cells-dev s3 cp s3://generate-cross-species/datasets/test/example.h5ad .

echo "Setup complete!" 
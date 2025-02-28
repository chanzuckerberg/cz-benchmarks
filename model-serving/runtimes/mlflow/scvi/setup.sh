#!/bin/bash

# Exit on error
set -e

ROOT_DIR=$(git rev-parse --show-toplevel)

echo "Copying data artifacts and example input file ..."

mkdir -p artifacts

for ORGANISM in homo_sapiens mus_musculus; do
    cp $ROOT_DIR/docker/scvi/hvg_names_${ORGANISM}.csv.gz artifacts/${ORGANISM}_hvg_names.csv.gz
    aws s3 --no-sign-request cp s3://cellxgene-contrib-public/models/scvi/2024-07-01/${ORGANISM}/model.pt artifacts/${ORGANISM}_model.pt
done
aws --profile single-cell-dev s3 cp s3://generate-cross-species/datasets/test/example_small.h5ad .

echo "Setup complete!" 
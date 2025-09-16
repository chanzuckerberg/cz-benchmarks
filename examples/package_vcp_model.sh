#!/bin/bash

# ==================================================
# Script: package_vcp_model.sh
#
# Overview:
#   This script helps you quickly set up and package scVI MLflow models.
#   It prepares everything you need to use these models with Python notebooks and czbenchmarks tasks.
#
# What this script does:
#   - Checks for required tools (Python 3.12 and AWS CLI).
#   - Clones the necessary GitHub repository if it isn't already present.
#   - Sets up a Python virtual environment and installs all dependencies.
#   - Downloads model files from AWS S3.
#   - Packages the downloaded models into an MLflow artifact.
#   - Saves the absolute path to the packaged model in a text file for easy access.
#
# What you need before running:
#   - Python 3.12 installed on your system.
#   - AWS CLI installed and configured with your credentials.
#
# Output:
#   - A directory containing the packaged MLflow model.
#   - A text file (mlflow_model_path.txt) with the full path to the model artifact.
#
# How to use:
#   1. Run this script.
#   2. Follow the next steps in the notebook: examples/czbenchmark_with_vcp_model.ipynb
#      (Prepare your dataset, run inference, and evaluate with czbenchmarks tasks.)
# ==================================================


echo "🚀 Starting scVI MLflow Model Setup and Packaging"
echo "=================================================="

# Set error handling
set -e

# Change below variables to package a different model
# Ensure the repository URL and directories match your setup
# You can also modify the model class and artifacts as needed.
REPO_URL="https://github.com/chanzuckerberg/vcp-model-pkg-client-tools.git"
REPO_DIR="vcp-model-pkg-client-tools"
MODEL_PKG_DIR="$REPO_DIR/examples/models/scvi/mlflow_pkg/"
VENV_NAME=".venv_mlflow_pkg"
MODEL_DATA_DIR="model_data"
MLFLOW_ARTIFACT_DIR="mlflow_model_artifact"

# Store original directory
ORIGINAL_DIR=$(pwd)

# Clone repository if it doesn't exist
if [ ! -d "$REPO_DIR" ]; then
    echo "🔄 Cloning repository..."
    git clone "$REPO_URL"
    echo "✅ Repository cloned successfully"
else
    echo "✅ Repository already exists"
fi

# Navigate to scvi_mlflow_pkg directory
cd "$MODEL_PKG_DIR"
echo "📁 Working directory: $(pwd)"

# # Modify requirements.txt for Mac compatibility. Uncomment below if needed.
# if [ -f "requirements.txt" ]; then
#     echo "🔄 Modifying requirements.txt for Mac compatibility..."
    
#     # Create backup
#     cp requirements.txt requirements.txt.backup
    
#     # Comment out NVIDIA-related libraries (in-place, cross-platform compatible)
#     sed -i.bak '/nvidia/ s/^/# /' requirements.txt
#     sed -i.bak '/cupy/ s/^/# /' requirements.txt
#     sed -i.bak '/torch+cu/ s/^/# /' requirements.txt
#     sed -i.bak '/cuda/ s/^/# /' requirements.txt
#     sed -i.bak '/triton/ s/^/# /' requirements.txt
    
#     # Remove temporary file
#     rm requirements.txt.bak
    
#     echo "✅ Modified requirements.txt for Mac compatibility"
# else
#     echo "❌ requirements.txt not found"
#     exit 1
# fi

# Check if python3.12 is available
if ! command -v python3.12 &> /dev/null; then
    echo "❌ python3.12 not found. Please install Python 3.12."
    echo "   You can download it from https://www.python.org/downloads/"
    exit 1
fi

# Create virtual environment
if [ ! -d "$VENV_NAME" ]; then
    echo "🔄 Creating virtual environment: $VENV_NAME"
    python3.12 -m venv "$VENV_NAME"
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment and install requirements
echo "🔄 Installing requirements in virtual environment..."
source "$VENV_NAME/bin/activate"
pip install --upgrade pip
pip install --no-deps -r requirements.txt
echo "✅ Requirements installed"

# Create model data directory
mkdir -p "$MODEL_DATA_DIR"

# Download model artifacts from S3
echo "🔄 Downloading model artifacts from S3..."
echo "   This may take several minutes depending on your internet connection..."

# Check if AWS CLI is available
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Please install AWS CLI and configure credentials."
    echo "   Installation: pip install awscli"
    echo "   Configuration: aws configure"
    exit 1
fi

# Download human model. Specific to SCVI models
# Adjust the S3 path as needed for your model version
echo "📥 Downloading human model artifacts..."
aws s3 sync s3://cz-benchmarks-data/models/v1/scvi_2023_12_15/homo_sapiens "$MODEL_DATA_DIR/homo_sapiens" || {
    echo "❌ Failed to download human model artifacts"
    echo "   Please ensure AWS CLI is configured with appropriate credentials"
    exit 1
}

# Download mouse model
echo "📥 Downloading mouse model artifacts..."
aws s3 sync s3://cz-benchmarks-data/models/v1/scvi_2023_12_15/mus_musculus "$MODEL_DATA_DIR/mus_musculus" || {
    echo "❌ Failed to download mouse model artifacts"
    exit 1
}

echo "✅ Model artifacts downloaded successfully"

# Verify model data structure
echo "📁 Model data structure:"
tree "$MODEL_DATA_DIR" 2>/dev/null || ls -la "$MODEL_DATA_DIR"/*/*

# Remove existing MLflow artifact directory
if [ -d "$MLFLOW_ARTIFACT_DIR" ]; then
    echo "🗑️  Removing existing MLflow artifact directory..."
    rm -rf "$MLFLOW_ARTIFACT_DIR"
fi

# Package the MLflow model
echo "🔄 Packaging MLflow model..."
python mlflow_packager.py \
    --model-class "model_code.scvi_mlflow_model:ScviMLflowModel" \
    --artifact "homo_sapiens=homo_sapiens" \
    --artifact "mus_musculus=mus_musculus" \
    --model-config-json '{"organism":"human"}' \
    --skip-inference

if [ $? -eq 0 ]; then
    echo "✅ MLflow model packaged successfully"
else
    echo "❌ Failed to package MLflow model"
    exit 1
fi

# Verify artifact structure
if [ -d "$MLFLOW_ARTIFACT_DIR" ]; then
    echo "📦 MLflow model artifact created successfully"
    echo "📁 Artifact structure:"
    tree "$MLFLOW_ARTIFACT_DIR" 2>/dev/null || find "$MLFLOW_ARTIFACT_DIR" -type f | head -20
    
    # Store the absolute path for Python notebooks
    MLFLOW_MODEL_PATH=$(pwd)/"$MLFLOW_ARTIFACT_DIR"
    echo "🎯 MLflow model ready at: $MLFLOW_MODEL_PATH"
    
    # Create a file with the model path for the Python notebooks to read
    echo "$MLFLOW_MODEL_PATH" > "$ORIGINAL_DIR/mlflow_model_path.txt"
    echo "💾 Model path saved to: $ORIGINAL_DIR/mlflow_model_path.txt"
    
else
    echo "❌ MLflow artifact directory not found"
    exit 1
fi

# Deactivate virtual environment
deactivate

# Return to original directory
cd "$ORIGINAL_DIR"

echo ""
echo "🎉 Setup completed successfully!"
echo "=================================================="
echo "📦 MLflow model packaged at: $MLFLOW_MODEL_PATH"
echo "📄 Model path saved to: mlflow_model_path.txt"
echo "🚀 Ready to proceed with the Python notebooks!"
echo ""
echo "Next steps:"
echo "  1. Run the dataset preparation cell"
echo "  2. Run MLflow model inference"
echo "  3. Evaluate with czbenchmarks tasks"
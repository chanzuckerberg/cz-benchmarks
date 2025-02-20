# Makefile for packaging and uploading model code to S3 for SageMaker

# Variables
S3_BUCKET ?= omar-data
MODEL_NAME ?= scvi
ARCHIVE_NAME ?= $(MODEL_NAME)_model_code.tar.gz
SOURCE_DIR ?= code/

.PHONY: all package upload clean

all: package upload

# Create a tar.gz archive of the source directory
package:
	tar -czvf $(ARCHIVE_NAME) $(SOURCE_DIR)

# Upload the archive to the specified S3 bucket
upload: package
	aws s3 cp $(ARCHIVE_NAME) s3://$(S3_BUCKET)/$(MODEL_NAME)/$(ARCHIVE_NAME) --profile virtual-cells-dev

# Clean up the archive file
clean:
	rm -f $(ARCHIVE_NAME)

sagemaker-deploy: upload
	python deploy.py
	make clean

sagemaker-test:
	python test.py

# Delete the endpoint and model
sagemaker-cleanup:
	python cleanup.py $(ENDPOINT_NAME) $(MODEL_NAME)

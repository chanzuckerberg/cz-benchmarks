This directory contains example notebooks and scripts demonstrating how to use the `czbenchmarks` library for benchmarking single-cell models. Each example is designed to showcase specific workflows, tasks, or utilities provided by the library. 


## **Notebook: using_czbenchmarks.ipynb**

### **Purpose**

This notebook provides a general introduction to using the `czbenchmarks` library. It guides users through loading datasets, running models, and evaluating results using standardized tasks and metrics.

### **Key Highlights**

- Loading single-cell datasets using `czbenchmarks`.
- Simulating model output for benchmarking tasks.
- Running tasks such as clustering, embedding quality, and label prediction.
- Comparing model performance with baseline methods.


## **Notebook: scvi_model_dev_workflow.ipynb**

### **Purpose**

This notebook demonstrates a developer workflow for evaluating and fine-tuning scVI models using `czbenchmarks`. It focuses on iterative model development and benchmarking during parameter tuning.

### **Key Highlights**

- Loading and preparing datasets for scVI models.
- Using pre-trained scVI model weights for inference.
- Fine-tuning scVI models with different hyperparameters.
- Evaluating clustering performance using metrics like Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI).
- Comparing pre-trained, fine-tuned, and baseline model performance.


## **Script: package_vcp_model.sh**

### **Purpose**

This script automates the process of packaging VCP models as MLflow artifacts. It prepares the environment, downloads model weights, and packages the model for use in benchmarking workflows.

### **Key Highlights**

- Cloning the required repository for scVI model packaging.
- Setting up a Python virtual environment and installing dependencies.
- Downloading model weights from AWS S3.
- Packaging the model as an MLflow artifact and saving the model path for use in notebooks.


## **Notebook: vcp_mlflow_model_benchmark.ipynb**

### **Purpose**

This notebook demonstrates how to benchmark a pre-trained scVI model packaged as an MLflow artifact using `czbenchmarks`. It includes steps for packaging the model, preparing datasets, running inference, and evaluating the model's performance on various tasks.

### **Key Highlights**

- Packaging a pre-trained scVI model as an MLflow artifact.
- Preparing datasets for benchmarking.
- Running inference through the MLflow model interface.
- Evaluating model embeddings using tasks like clustering, embedding quality, and metadata label prediction.
- Comparing results with baselines.


### **How to Use These Examples**

1. **Prerequisites**: Ensure you have Python 3.12, AWS CLI, and other dependencies installed. Follow the setup instructions in the notebooks or scripts.
2. **Run the Examples**: Open the notebooks in Jupyter or VS Code and execute the cells step-by-step.
3. **Adapt for Your Workflow**: Replace the provided models or datasets with your own to benchmark custom workflows.

For more details, refer to the comments and instructions within each notebook or script.
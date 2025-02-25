from pathlib import Path
import sys
import anndata as ad
import mlflow

from mlflow_scvi import MLflowSCVI
import argparse
    
    
if __name__ == "__main__":
    model = mlflow.pyfunc.load_model("runtime")
    print(model)

    parser = argparse.ArgumentParser(description="Run SCVI model predictions.")
    parser.add_argument("--input-h5ad", help="Path to the input .h5ad file")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()
    

    print("\nRunning prediction...\n")

    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        # For testing the internal prediction method, w/o mlflow input or context handling
        adata = ad.read_h5ad(args.input_h5ad)
        print(adata)
        # input_data = pickle.dumps(adata)
        result = MLflowSCVI()._predict(adata, 
                                       "artifacts/homo_sapiens/hvg_names.csv.gz",
                                       Path("artifacts/homo_sapiens"))
        pass
    else:
        result = model.predict(args.input_h5ad, params=dict(organism="homo_sapiens"))
        
    print(result)
    print(result.shape)
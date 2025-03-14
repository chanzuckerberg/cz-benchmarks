import argparse
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np
import scipy.sparse
from geneformer import EmbExtractor, TranscriptomeTokenizer
from omegaconf import OmegaConf
from datasets import load_from_disk, Sequence, Value

from czbenchmarks.datasets import BaseDataset, DataType
from czbenchmarks.models.implementations.base_model_implementation import (
    BaseModelImplementation,
)
from czbenchmarks.models.validators.geneformer import GeneformerValidator
from czbenchmarks.utils import sync_s3_to_local


class Geneformer(GeneformerValidator, BaseModelImplementation):
    def __init__(self):
        super().__init__()
        self.args = self.parse_args()
        self.config = OmegaConf.load("config.yaml")
        self.selected_model = self.config.models[self.args.model_variant]
        self.token_config = self.selected_model.token_config

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_variant", type=str, default="gf_12L_30M")
        return parser.parse_args()

    def get_model_weights_subdir(self, _dataset: BaseDataset) -> str:
        assert (
            self.args.model_variant in self.config.models
        ), f"Model {self.args.model_variant} not found in config"
        return self.args.model_variant

    def _download_model_weights(self, _dataset: BaseDataset):
        model_uri = self.selected_model.model_uri
        Path(self.model_weights_dir).mkdir(parents=True, exist_ok=True)

        bucket, key = model_uri.split("/")[2], "/".join(model_uri.split("/")[3:])
        sync_s3_to_local(bucket, key, self.model_weights_dir)

    def run_model(self, dataset: BaseDataset):
        seq_len = self.token_config.input_size

        # Check input data quality
        X = dataset.adata.X
        if scipy.sparse.issparse(X):
            # For sparse matrix, only check non-zero elements
            print("Input is sparse matrix, checking non-zero elements for NaN...")
            X_data = X.data
            if np.isnan(X_data).any():
                print("Warning: Input data contains NaN values in non-zero elements")
                print("Number of NaN values:", np.isnan(X_data).sum())
        else:
            # For dense matrix
            if np.isnan(X).any():
                print("Warning: Input data contains NaN values")
                print("NaN locations in input:", np.where(np.isnan(X)))

        # Add necessary metadata
        dataset.adata.obs["cell_idx"] = np.arange(len(dataset.adata.obs))
        if "n_counts" not in dataset.adata.obs.columns:
            n_counts = np.asarray(dataset.adata.X.sum(axis=1)).flatten()
            if np.isnan(n_counts).any():
                print("Warning: NaN values in n_counts calculation")
            dataset.adata.obs["n_counts"] = n_counts

        # Clean gene IDs by removing version numbers
        print("\nCleaning gene IDs...")
        print("Before cleaning - sample IDs:", dataset.adata.var["ensembl_id"].head())
        # Update ensembl_id column with cleaned values
        dataset.adata.var["ensembl_id"] = (
            dataset.adata.var["ensembl_id"].str.split('.').str[0]
        )
        print("After cleaning - sample IDs:", dataset.adata.var["ensembl_id"].head())

        # Save dataset to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp_file:
            temp_path = Path(tmp_file.name)
        dataset.adata.write_h5ad(temp_path)

        # Initialize and run tokenizer
        dataset_dir = Path("dataset")
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Print gene information before tokenization
        print("\nGene Information:")
        print("Number of genes:", dataset.adata.n_vars)
        print("Available var columns:", dataset.adata.var.columns.tolist())
        print("Sample of gene IDs:", dataset.adata.var["ensembl_id"].head())

        # Create a minimal gene mapping that maps each ENSG ID to itself
        print("\nCreating minimal gene mapping...")
        our_genes = set(dataset.adata.var["ensembl_id"].str.split('.').str[0])
        minimal_gene_map = {g: g for g in our_genes}  # Each gene maps to itself

        # Save the minimal mapping to a temporary file
        minimal_map_path = Path("minimal_gene_map.pkl")
        with open(minimal_map_path, 'wb') as f:
            pickle.dump(minimal_gene_map, f)

        print(f"Created minimal gene mapping with {len(minimal_gene_map)} genes")
        print("Sample mappings:", list(minimal_gene_map.items())[:5])
        print("\nFirst few rows of var dataframe:")
        print(dataset.adata.var.head())

        # Verify data format requirements
        print("\nVerifying data format requirements:")
        print("1. ensembl_id in var:", "ensembl_id" in dataset.adata.var.columns)
        print("2. n_counts in obs:", "n_counts" in dataset.adata.obs.columns)

        # Check gene ID format
        print("\nGene ID format check:")
        sample_genes = dataset.adata.var["ensembl_id"].head(10)
        print("Original genes:", sample_genes.tolist())
        print("Cleaned genes:", [g.split('.')[0] for g in sample_genes])

        # Load token dictionary to verify gene format matches
        token_dict_path = Path(self.token_config.token_dictionary_file)
        with open(token_dict_path, 'rb') as f:
            token_dict = pickle.load(f)

        # Check if our cleaned genes are in token dictionary
        cleaned_genes = [g.split('.')[0] for g in sample_genes]
        print("\nToken dictionary check:")
        for gene in cleaned_genes:
            print(f"{gene}: {'✓' if gene in token_dict else '✗'}")
            if gene in token_dict:
                print(f"  Token ID: {token_dict[gene]}")

        tk = TranscriptomeTokenizer(
            custom_attr_name_dict={"cell_idx": "cell_idx"},
            nproc=4,
            gene_median_file=str(Path(self.token_config.gene_median_file)),
            token_dictionary_file=str(Path(self.token_config.token_dictionary_file)),
            gene_mapping_file=str(minimal_map_path),  # Use our minimal mapping
            special_token=(seq_len != 2048),
            model_input_size=seq_len,
        )

        # Print tokenizer configuration
        print("\nTokenizer Configuration:")
        print("Model input size:", seq_len)
        print("Special token:", (seq_len != 2048))
        print("Gene median file:", self.token_config.gene_median_file)
        print("Token dictionary file:", self.token_config.token_dictionary_file)
        print("Using minimal gene mapping file")

        # Use the correct directory where the temp file is saved
        tk.tokenize_data(
            str(temp_path.parent),
            str(dataset_dir),
            "tokenized_dataset",
            file_format="h5ad",
        )

        # Check tokenized data quality
        dataset_name = "tokenized_dataset.dataset"
        tokenized_dataset_path = dataset_dir / dataset_name
        tokenized_dataset = load_from_disk(str(tokenized_dataset_path))

        print("\nTokenized Data Quality Check:")
        print("Number of sequences:", len(tokenized_dataset))
        print("Dataset features:", tokenized_dataset.features)
        print("Dataset columns:", tokenized_dataset.column_names)

        # Inspect first sequence in detail
        first_seq = tokenized_dataset[0]
        print("\nFirst sequence details:")
        for key, value in first_seq.items():
            print(f"{key}:", type(value), "shape/len:", np.array(value).shape)
            print(f"{key} content:", value)

        # Now check actual sequence length
        input_ids = np.array(tokenized_dataset[0]["input_ids"])
        if input_ids.ndim != 1:
            print(f"Warning: input_ids has unexpected shape: {input_ids.shape}")
        print("Actual sequence length:", input_ids.shape)

        # Validate sequence length
        if len(tokenized_dataset[0]["input_ids"]) < 10:  # arbitrary minimum length
            seq_len = len(tokenized_dataset[0]["input_ids"])
            raise ValueError(
                f"Tokenized sequences are too short (length={seq_len}). "
                "This suggests a problem with the tokenization process. "
                "Expected longer sequences for gene expression data."
            )

        # Check dtype and convert if needed
        input_ids_dtype = np.array(tokenized_dataset["input_ids"][0]).dtype
        if np.issubdtype(input_ids_dtype, np.floating):
            # Convert only if input_ids are floats
            new_features = tokenized_dataset.features.copy()
            new_features["input_ids"] = Sequence(Value("int64"))
            tokenized_dataset = tokenized_dataset.cast(new_features)

            # Change dataset name to avoid overwriting the original dataset
            dataset_name = "tokenized_dataset_int.dataset"
            tokenized_dataset_path = dataset_dir / dataset_name
            tokenized_dataset.save_to_disk(str(tokenized_dataset_path))

        del tokenized_dataset  # Free memory

        # Extract embeddings with detailed monitoring
        embex = EmbExtractor(
            model_type="Pretrained",
            emb_layer=-1,
            emb_mode="cell",
            forward_batch_size=32,
            nproc=4,
            token_dictionary_file=str(Path(self.token_config.token_dictionary_file)),
            max_ncells=None,
            emb_label=["cell_idx"],
        )

        embs = embex.extract_embs(
            model_directory=self.model_weights_dir,
            input_data_file=str(tokenized_dataset_path),  # Use dataset_name variable
            output_directory=".",
            output_prefix="geneformer",
            cell_state=None,
            output_torch_embs=False,
        )

        # Check embedding quality before sorting
        if embs.isna().any().any():
            print("\nEmbedding Quality Report:")
            print("Total embeddings:", len(embs))
            print("NaN counts per dimension:", embs.isna().sum())
            print("Rows with NaNs:", len(embs[embs.isna().any(axis=1)]))
            print("Sample of problematic rows:", embs[embs.isna().any(axis=1)].head())

        # Sort embeddings and store output
        embs = embs.sort_values("cell_idx").drop(columns=["cell_idx"])

        # Convert to numpy array and handle NaN values
        emb_array = embs.to_numpy()
        if np.isnan(emb_array).any():
            nan_count = np.isnan(emb_array).sum()
            total_values = emb_array.size
            print(f"\nFound {nan_count} NaN values out of {total_values} total values")
            print(f"NaN percentage: {(nan_count/total_values)*100:.2f}%")
            emb_array = np.nan_to_num(emb_array, nan=0.0)

        # Verify no NaN values remain
        assert not np.isnan(
            emb_array
        ).any(), "NaN values still present in embeddings after handling"

        dataset.set_output(self.model_type, DataType.EMBEDDING, emb_array)

        # Cleanup
        temp_path.unlink(missing_ok=True)
        minimal_map_path.unlink(missing_ok=True)  # Clean up our temporary mapping file
        shutil.rmtree(dataset_dir, ignore_errors=True)


if __name__ == "__main__":
    Geneformer().run()

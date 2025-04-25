import argparse
import os
import logging
from typing import Tuple, List

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.cuda.amp import autocast
from modelgenerator.tasks import Embed

from czbenchmarks.datasets import BaseDataset, DataType
from czbenchmarks.models.implementations.base_model_implementation import BaseModelImplementation
from czbenchmarks.models.validators import BaseSingleCellValidator
from czbenchmarks.models.types import ModelType
from czbenchmarks.datasets.single_cell import Organism
from czbenchmarks.datasets.utils import load_dataset

import scanpy as sc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def align_cells_to_model(
    adata: ad.AnnData,
    model_gene_file: str,
    feature_col: str = "feature_name",
    model_csv_gene_column: str = "gene_name"
) -> Tuple[ad.AnnData, np.ndarray]:
    """
    Align an AnnData object's genes to the model's gene list.
    """
    # Step 1: Map feature_col to var_names
    if feature_col in adata.var.columns:
        adata.var_names = adata.var[feature_col].astype(str)  # Ensure strings
    else:
        logger.error("Missing '%s' in adata.var; cannot align genes.", feature_col)
        raise KeyError(f"Required column '{feature_col}' not found in adata.var.")

    # Step 2: Load model genes from the specified column in the TSV file
    try:
        model_genes_df = pd.read_csv(model_gene_file, sep="\t", usecols=[model_csv_gene_column])
        model_genes = model_genes_df[model_csv_gene_column].tolist()
    except Exception as e:
        logger.error("Error loading model gene file '%s': %s", model_gene_file, str(e))
        raise

    # Step 3: Prepare index mappings for input and model genes
    input_genes = list(adata.var_names)
    input_gene_to_index = {gene: idx for idx, gene in enumerate(input_genes)}
    model_gene_to_index = {gene: idx for idx, gene in enumerate(model_genes)}
    common_genes = [gene for gene in model_genes if gene in input_gene_to_index]

    logger.info("Found %d / %d model genes in input AnnData.", len(common_genes), len(model_genes))
    if not common_genes:
        raise ValueError("No overlap between input AnnData genes and model gene set.")

    # Step 4: Extract the numeric matrix from AnnData
    if sp.issparse(adata.X):
        input_matrix = adata.X.toarray()
    elif isinstance(adata.X, np.ndarray):
        input_matrix = adata.X
    else:
        raise TypeError(f"Unsupported AnnData.X type: {type(adata.X)}")

    # Step 5: Build the aligned matrix with zero-padding for missing genes
    aligned_matrix = np.zeros((input_matrix.shape[0], len(model_genes)), dtype=input_matrix.dtype)
    for gene in common_genes:
        aligned_matrix[:, model_gene_to_index[gene]] = input_matrix[:, input_gene_to_index[gene]]

    aligned_adata = ad.AnnData(
        X=sp.csr_matrix(aligned_matrix),
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=model_genes)
    )

    # Step 6: Create a boolean mask indicating which model genes are present
    gene_presence_mask = np.array([gene in common_genes for gene in model_genes], dtype=bool)

    return aligned_adata, gene_presence_mask


def get_gene_embeddings_from_model(
    model: torch.nn.Module,
    aligned_adata: ad.AnnData,
    mask: np.ndarray,
    batch_size: int,
    device: torch.device,
    use_amp: bool = True
) -> np.ndarray:
    """
    Batch-wise inference: apply model, mask genes, and average embeddings.
    Returns a [cells x embedding_dim] NumPy array.
    """
    # Prepare dense data
    if sp.issparse(aligned_adata.X):
        data = aligned_adata.X.toarray()
    else:
        data = aligned_adata.X

    num_cells, num_genes = data.shape
    embeddings_list: List[np.ndarray] = []

    model.eval()
    model = model.half().to(device)

    # Process data in batches
    for i in range(0, num_cells, batch_size):
        end = min(i + batch_size, num_cells)
        logger.info("Processing batch %d to %d", i, end)

        batch_np = data[i:end]
        batch_tensor = torch.from_numpy(batch_np).to(device=device, dtype=torch.float16)

        # Prepare attention mask
        attention_mask = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0).expand(batch_tensor.size(0), -1)

        # Perform inference
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            try:
                out = model(batch_tensor, attention_mask=attention_mask)  # Pass attention_mask to the model
            except Exception as e:
                logger.error(f"Error during model execution for batch {i} to {end}: {e}")
                raise

        # Validate gene dimension
        if out.shape[1] != mask.shape[0]:
            raise ValueError(
                f"Model output gene dim ({out.shape[1]}) != mask length ({mask.shape[0]})"
            )

        # Apply mask to the output
        out_masked = out[:, mask]  # Select only valid genes based on the mask

        # Collapse genes by averaging
        emb2d = out_masked.mean(dim=1)  # [batch x dim]
        embeddings_list.append(emb2d.detach().cpu().numpy())  # Detach before converting to NumPy

    # Concatenate and sanitize
    embeddings = np.concatenate(embeddings_list, axis=0)
    if np.isnan(embeddings).any():
        logger.warning("NaN values detected; replacing with zero.")
        embeddings = np.nan_to_num(embeddings)
    return embeddings


class AIDOValidator(BaseSingleCellValidator):
    available_organisms = [Organism.HUMAN]
    required_obs_keys = []
    required_var_keys = ['feature_name']
    model_type = ModelType.AIDO

    @property
    def inputs(self):
        return {DataType.ANNDATA}

    @property
    def outputs(self):
        return {DataType.EMBEDDING}


class AIDO(AIDOValidator, BaseModelImplementation):
    """AIDO single-cell embedding model."""
    def get_model_weights_subdir(self, dataset: BaseDataset) -> str:
        pass

    def _download_model_weights(self, dataset: BaseDataset):
        pass

    def parse_args(self):
        parser = argparse.ArgumentParser(description="AIDO Model Arguments")
        parser.add_argument(
            "--batch_size",
            type=int,
            default=16,
            help="Batch size for processing data (default: 2)"
        )
        parser.add_argument(
            "--model_variant",
            type=str,
            choices=["aido_cell_3m", "aido_cell_10m" "aido_cell_100m"],
            default="aido_cell_3m",
            help="Model variant to use (default: aido_cell_3m)"
        )
        args = parser.parse_args()
        return args

    def run_model(self, dataset: BaseDataset):
        try:
            args = self.parse_args()
            batch_size = args.batch_size
            variant = args.model_variant
            adata: ad.AnnData = dataset.adata

            logger.info(
                "AIDO: Running %s on %d cells × %d genes",
                variant, *adata.shape
            )

            # Align genes
            gene_file = os.path.join(
                os.path.dirname(__file__),
                "OS_scRNA_gene_index.19264.tsv"
            )
            aligned_adata, mask = align_cells_to_model(
                adata, model_gene_file=gene_file
            )

            # Load model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = Embed.from_config({
                "model.backbone": variant,
                "model.batch_size": batch_size
            }).backbone
            model = model.eval().to(device).to(torch.float16)

            # Batch inference with dynamic batch size adjustment
            while True:
                try:
                    embeddings = get_gene_embeddings_from_model(
                        model=model,
                        aligned_adata=aligned_adata,
                        mask=mask,
                        batch_size=batch_size,
                        device=device
                    )
                    break  # Exit loop if successful
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"Out of memory with batch size {batch_size}. Reducing batch size by half.")
                    batch_size = max(1, batch_size // 2)  # Reduce batch size by half
                    if batch_size == 1:
                        logger.error("Batch size reduced to 1, but still out of memory.")
                        raise

            # Set output
            dataset.set_output(
                self.model_type,
                DataType.EMBEDDING,
                embeddings
            )
            logger.info("AIDO: Completed embedding of %d cells.", embeddings.shape[0])

        except Exception:
            logger.exception("Error in AIDO run_model")
            raise


if __name__ == "__main__":
    AIDO().run()


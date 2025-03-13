import argparse
import logging
import pathlib
from glob import glob

import numpy as np
import pandas as pd
import torch
from gears import PertData
from omegaconf import OmegaConf

# utils.data_loading is a function in https://github.com/czi-ai/scGenePT/tree/main
from utils.data_loading import load_trained_scgenept_model

from czbenchmarks.datasets import BaseDataset, DataType
from czbenchmarks.models.implementations.base_model_implementation import (
    BaseModelImplementation,
)
from czbenchmarks.models.validators.scgenept import ScGenePTValidator
from czbenchmarks.utils import download_s3_file, sync_s3_to_local

logger = logging.getLogger(__name__)


def load_dataloader(
    dataset_name, data_dir, batch_size, val_batch_size, split="simulation"
):
    pert_data = PertData(f"{data_dir}/")
    pert_data.load(data_name=dataset_name)
    pert_data.prepare_split(split=split, seed=1)
    pert_data.get_dataloader(batch_size=batch_size, test_batch_size=val_batch_size)
    return pert_data


class ScGenePT(ScGenePTValidator, BaseModelImplementation):
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_variant", type=str, default="scgenept_go_c")
        parser.add_argument("--gene_pert", type=str, default="CEBPB+ctrl")
        parser.add_argument("--dataset_name", type=str, default="adamson")
        parser.add_argument("--chunk_size", type=int, default=512)
        args = parser.parse_args()
        return args

    def get_model_weights_subdir(self, _dataset: BaseDataset) -> str:
        args = self.parse_args()
        config = OmegaConf.load("config.yaml")
        assert (
            f"{args.model_variant}__{args.dataset_name}" in config.models
        ), f"Model {args.model_variant}__{args.dataset_name} not found in config"
        return f"{args.model_variant}/{args.dataset_name}"

    def _download_model_weights(self, _dataset: BaseDataset):
        config = OmegaConf.load("config.yaml")
        args = self.parse_args()

        # Sync the finetuned model weights from S3 to the local model weights directory
        model_uri = config.models[f"{args.model_variant}__{args.dataset_name}"]

        # Create all parent directories
        pathlib.Path(self.model_weights_dir).mkdir(parents=True, exist_ok=True)

        bucket = model_uri.split("/")[2]
        key = "/".join(model_uri.split("/")[3:])

        sync_s3_to_local(bucket, key, self.model_weights_dir)
        logger.info(
            f"Downloaded model weights from {model_uri} to " f"{self.model_weights_dir}"
        )

        # Copy the vocab.json file from S3 to local model weights directory
        vocab_uri = config.models["vocab_uri"]
        vocab_key = "/".join(vocab_uri.split("/")[3:])

        vocab_dir = (
            pathlib.Path(self.model_weights_dir).parent.parent / "pretrained" / "scgpt"
        )
        vocab_dir.mkdir(parents=True, exist_ok=True)
        vocab_file = vocab_dir / "vocab.json"

        download_s3_file(bucket, vocab_key, str(vocab_file))
        logger.info(f"Downloaded vocab.json from {vocab_uri} to {vocab_file}")

        # Copy the gene_embeddings directory from S3 to local model weights directory
        gene_embeddings_uri = config.models["gene_embeddings_uri"]
        gene_embeddings_key = "/".join(gene_embeddings_uri.split("/")[3:])
        gene_embeddings_dir = (
            pathlib.Path(self.model_weights_dir).parent.parent / "gene_embeddings"
        )
        gene_embeddings_dir.mkdir(parents=True, exist_ok=True)
        sync_s3_to_local(bucket, gene_embeddings_key, str(gene_embeddings_dir))
        logger.info(
            f"Downloaded gene_embeddings from {gene_embeddings_uri} "
            f"to {gene_embeddings_dir}"
        )

    def run_model(self, dataset: BaseDataset):
        adata = dataset.adata
        adata.var["gene_name"] = adata.var["feature_name"]

        args = self.parse_args()
        dataset_name = args.dataset_name
        batch_size = 64
        eval_batch_size = 64

        pert_data_dir = (
            f"{str(pathlib.Path(self.model_weights_dir).parent.parent)}/data"
        )
        pathlib.Path(pert_data_dir).mkdir(parents=True, exist_ok=True)
        pert_data = load_dataloader(
            dataset_name,
            pert_data_dir,
            batch_size,
            eval_batch_size,
            split="simulation",
        )
        ref_adata = pert_data.adata

        adata = adata[
            :, [i for i in ref_adata.var_names if i in adata.var_names]
        ].copy()
        ref_adata = ref_adata[
            :, [i for i in ref_adata.var_names if i in adata.var_names]
        ].copy()

        model_filename = glob(f"{self.model_weights_dir}/*.pt")[0]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, gene_ids = load_trained_scgenept_model(
            ref_adata,
            f"{args.model_variant}_gpt_concat",
            str(pathlib.Path(self.model_weights_dir).parent.parent) + "/",
            model_filename,
            device,
        )

        gene_names = adata.var["gene_name"].to_list()
        gene_pert = args.gene_pert
        chunk_size = args.chunk_size
        all_preds = []

        num_chunks = (adata.shape[0] + chunk_size - 1) // chunk_size
        for i in range(num_chunks):
            logger.info(f"Predicting perturbations for chunk {i + 1} of {num_chunks}")
            chunk = adata[i * chunk_size : (i + 1) * chunk_size]
            preds = model.pred_perturb_from_ctrl(
                chunk,
                gene_pert,
                gene_names,
                device,
                gene_ids,
                pool_size=None,
                return_mean=False,
            ).squeeze()
            all_preds.append(preds)

        dataset.set_output(
            self.model_type,
            DataType.PERTURBATION_PRED,
            (
                gene_pert,
                pd.DataFrame(
                    data=np.concatenate(all_preds, axis=0),
                    index=adata.obs_names,
                    columns=adata.var_names.to_list(),
                ),
            ),
        )


if __name__ == "__main__":
    ScGenePT().run()

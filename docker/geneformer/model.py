from pathlib import Path

from geneformer import EmbExtractor, TranscriptomeTokenizer
from omegaconf import OmegaConf

from czibench.datasets.sc import SingleCellDataset
from czibench.datasets.types import Organism
from czibench.models.sc import BaseSingleCell
from czibench.utils import sync_s3_to_local


class Geneformer(BaseSingleCell):
    available_organisms = [Organism.HUMAN]
    required_obs_keys = []
    required_var_keys = ["feature_id"]

    @classmethod
    def _validate_model_requirements(cls, dataset: SingleCellDataset):
        missing_keys = [
            key for key in cls.required_var_keys if key not in dataset.adata.var.columns
        ]
        if missing_keys:
            raise ValueError(f"Missing required var keys: {missing_keys}")

    def parse_args(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name", type=str, default="gf_12L_30M")
        args = parser.parse_args()
        return args
    
    def get_model_weights_subdir(self) -> str:
        args = self.parse_args()
        config = OmegaConf.load("config.yaml")
        assert (
            f"{args.model_name}" in config.models
        ), f"Model {args.model_name} not found in config"
        return args.model_name


    def _download_model_weights(self):
        config = OmegaConf.load("config.yaml")
        args = self.parse_args()
        selected_model = config.models[args.model_name]
        model_uri = selected_model.model_uri

        Path(self.model_weights_dir).mkdir(exist_ok=True)

        bucket = model_uri.split("/")[2]
        key = "/".join(model_uri.split("/")[3:])

        sync_s3_to_local(bucket, key, self.model_weights_dir)

    def run_model(self):
        config = OmegaConf.load("config.yaml")
        args = self.parse_args()
        selected_model = config.models[args.model_name]
        token_config = selected_model.token_config
        seq_len = token_config.input_size

        # Add cell index as metadata to track order
        self.data.adata.obs['cell_idx'] = range(len(self.data.adata.obs))

        # Add n_counts if not present
        if "n_counts" not in self.data.adata.obs.columns:
            self.data.adata.obs["n_counts"] = self.data.adata.X.sum(axis=1)

        # Save adata to temp file
        temp_path = Path("temp_dataset.h5ad")
        self.data.adata.write_h5ad(temp_path)

        # Initialize tokenizer with cell_idx tracking
        tk = TranscriptomeTokenizer(
            custom_attr_name_dict={"cell_idx": "cell_idx"},  # Track cell order
            nproc=4,
            gene_median_file=str(Path(token_config.gene_median_file)),
            token_dictionary_file=str(Path(token_config.token_dictionary_file)),
            gene_mapping_file=str(Path(token_config.ensembl_mapping_file)),
            special_token=(seq_len != 2048),
            model_input_size=seq_len,
        )

        # Create dataset directory
        dataset_dir = Path("dataset")
        dataset_dir.mkdir(exist_ok=True)

        # Tokenize data
        tk.tokenize_data(".", str(dataset_dir), "tokenized_dataset", file_format="h5ad")

        # Extract embeddings with cell_idx label
        embex = EmbExtractor(
            model_type="Pretrained",
            emb_layer=-1,
            emb_mode="cell",
            forward_batch_size=32,
            nproc=4,
            token_dictionary_file=str(Path(token_config.token_dictionary_file)),
            max_ncells=None,
            emb_label=['cell_idx']  # Include cell_idx in output
        )

        # Get embeddings
        embs = embex.extract_embs(
            model_directory=self.model_weights_dir,
            input_data_file=str(dataset_dir / "tokenized_dataset.dataset"),
            output_directory=".",
            output_prefix="geneformer",
            cell_state=None,
            output_torch_embs=False,
        )

        # Sort embeddings by cell_idx to restore original order
        embs = embs.sort_values('cell_idx')
        
        # Remove the cell_idx column and convert to numpy array
        embs = embs.drop('cell_idx', axis=1)
        self.data.output_embedding = embs.values

        # Cleanup
        temp_path.unlink()
        import shutil
        shutil.rmtree(dataset_dir)


if __name__ == "__main__":
    Geneformer().run()

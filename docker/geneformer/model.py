import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
from geneformer import EmbExtractor, TranscriptomeTokenizer
from omegaconf import OmegaConf
from datasets import load_from_disk, Sequence, Value

from czbenchmarks.datasets import BaseDataset, DataType
from czbenchmarks.models.implementations.base_model_implementation import BaseModelImplementation
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
        assert self.args.model_variant in self.config.models, f"Model {self.args.model_variant} not found in config"
        return self.args.model_variant

    def _download_model_weights(self, _dataset: BaseDataset):
        model_uri = self.selected_model.model_uri
        Path(self.model_weights_dir).mkdir(parents=True, exist_ok=True)

        bucket, key = model_uri.split("/")[2], "/".join(model_uri.split("/")[3:])
        sync_s3_to_local(bucket, key, self.model_weights_dir)

    def run_model(self, dataset: BaseDataset):
        seq_len = self.token_config.input_size

        # Add necessary metadata
        dataset.adata.obs["cell_idx"] = np.arange(len(dataset.adata.obs))
        if "n_counts" not in dataset.adata.obs.columns:
            dataset.adata.obs["n_counts"] = np.asarray(dataset.adata.X.sum(axis=1)).flatten()

        # Save dataset to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp_file:
            temp_path = Path(tmp_file.name)
        dataset.adata.write_h5ad(temp_path)

        # Initialize and run tokenizer
        dataset_dir = Path("dataset")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        tk = TranscriptomeTokenizer(
            custom_attr_name_dict={"cell_idx": "cell_idx"},
            nproc=4,
            gene_median_file=str(Path(self.token_config.gene_median_file)),
            token_dictionary_file=str(Path(self.token_config.token_dictionary_file)),
            gene_mapping_file=str(Path(self.token_config.ensembl_mapping_file)),
            special_token=(seq_len != 2048),
            model_input_size=seq_len,
        )

        # Use the correct directory where the temp file is saved
        tk.tokenize_data(str(temp_path.parent), str(dataset_dir), "tokenized_dataset", file_format="h5ad")

        # Set dataset name to default
        dataset_name = "tokenized_dataset.dataset"
        tokenized_dataset_path = dataset_dir / dataset_name
        tokenized_dataset = load_from_disk(str(tokenized_dataset_path))

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

        # Extract embeddings
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

        # Sort embeddings and store output
        embs = embs.sort_values("cell_idx").drop(columns=["cell_idx"])
        dataset.set_output(self.model_type, DataType.EMBEDDING, embs.to_numpy())

        # Cleanup
        temp_path.unlink(missing_ok=True)
        shutil.rmtree(dataset_dir, ignore_errors=True)


if __name__ == "__main__":
    Geneformer().run()

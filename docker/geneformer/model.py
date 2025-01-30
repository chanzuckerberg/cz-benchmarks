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

    def get_model_weights_subdir(self) -> str:
        config = OmegaConf.load("config.yaml")
        selected_model = config.models[config.default_model]
        return selected_model.model_name

    def _download_model_weights(self):
        config = OmegaConf.load("config.yaml")
        selected_model = config.models[config.default_model]
        model_uri = selected_model.model_uri

        Path(self.model_weights_dir).mkdir(exist_ok=True)

        bucket = model_uri.split("/")[2]
        key = "/".join(model_uri.split("/")[3:])

        sync_s3_to_local(bucket, key, self.model_weights_dir)

    def run_model(self):
        config = OmegaConf.load("config.yaml")
        selected_model = config.models[config.default_model]
        token_config = selected_model.token_config
        seq_len = token_config.input_size

        # Setup tokenizer
        tk = TranscriptomeTokenizer(
            {},  # No metadata mapping needed
            nproc=16,
            gene_median_file=str(Path(token_config.gene_median_file)),
            token_dictionary_file=str(Path(token_config.token_dictionary_file)),
            gene_mapping_file=str(Path(token_config.ensembl_mapping_file)),
            special_token=(seq_len != 2048),  # True for 95M models, False for 30M
            model_input_size=seq_len,
        )

        # Extract embeddings
        embex = EmbExtractor(
            model_type="Pretrained",
            emb_layer=-1,
            emb_mode="cell",
            forward_batch_size=200 if seq_len == 2048 else 100,
            nproc=16,
            token_dictionary_file=str(Path(token_config.token_dictionary_file)),
            max_ncells=None,
        )

        # Get embeddings
        embs = embex.extract_embs(
            model_directory=self.model_weights_dir,
            input_data=self.data.adata,
            tokenizer=tk,
            return_tensor=True,
        )

        # Store embeddings
        self.data.output_embedding = embs.cpu().numpy()


if __name__ == "__main__":
    Geneformer().run()
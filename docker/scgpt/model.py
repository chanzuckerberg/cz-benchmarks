import pathlib
import scgpt as scg
from omegaconf import OmegaConf
import argparse

from czibench.models.validators.scgpt import ScGPTValidator
from czibench.models.base import BaseModelImplementation
from czibench.utils import sync_s3_to_local
from czibench.datasets.types import DataType


class ScGPT(ScGPTValidator, BaseModelImplementation):
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name", type=str, default="human")
        args = parser.parse_args()
        return args

    def get_model_weights_subdir(self) -> str:
        args = self.parse_args()
        config = OmegaConf.load("config.yaml")
        selected_model = config.models[args.model_name]
        model_name = selected_model.model_name
        return model_name

    def _download_model_weights(self):
        config = OmegaConf.load("config.yaml")
        args = self.parse_args()
        selected_model = config.models[args.model_name]
        model_uri = selected_model.model_uri

        pathlib.Path(self.model_weights_dir).mkdir(exist_ok=True)

        bucket = model_uri.split("/")[2]
        key = "/".join(model_uri.split("/")[3:])

        sync_s3_to_local(bucket, key, self.model_weights_dir)

    def run_model(self):
        adata = self.data.adata
        adata.var["gene_name"] = adata.var["gene_symbol"]
        ref_embed_adata = scg.tasks.embed_data(
            adata,
            model_dir=self.model_weights_dir,
            gene_col="gene_name",
            batch_size=32,
        )
        self.set_output(DataType.EMBEDDING, ref_embed_adata.obsm["X_scGPT"])


if __name__ == "__main__":
    ScGPT().run()

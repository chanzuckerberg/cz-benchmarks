import pathlib
from datetime import datetime

import anndata as ad
import scgpt as scg
from omegaconf import OmegaConf
from czibench.models.sc import ScGPTValidator


class ScGPT(ScGPTValidator):

    def run_model(self):
        """
        Required: Implement your model's inference logic here.
        Access input data via self.data.adata
        Set output embedding via self.data.output_embedding
        """
        config = OmegaConf.load("config.yaml")
        adata = self.data.adata
        adata.var["gene_name"] = adata.var["feature_name"] # gene_symbol is for the bladder dataset
        model_name = config.model.model_name
        ref_embed_adata = scg.tasks.embed_data(
            adata,
            model_dir=pathlib.Path(config.paths.model_dir) / model_name,
            gene_col="gene_name",
            batch_size=32,
        )

        self.data.output_embedding = ref_embed_adata.obsm["X_scGPT"]

if __name__ == "__main__":
    ScGPT().run()

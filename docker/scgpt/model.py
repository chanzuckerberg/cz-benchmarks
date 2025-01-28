import pathlib
from datetime import datetime

import anndata as ad
import scgpt as scg
from omegaconf import OmegaConf
from czibench.models.sc import ScGPTValidator


class ScGPT(ScGPTValidator):

    def run_model(self):
        config = OmegaConf.load("config.yaml")
        adata = self.data.adata
<<<<<<< Updated upstream
        adata.var["gene_name"] = adata.var["gene_symbol"]
=======
        adata.var["gene_name"] = adata.var["feature_name"]  # feature_name is for the example.h5ad dataset
>>>>>>> Stashed changes
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

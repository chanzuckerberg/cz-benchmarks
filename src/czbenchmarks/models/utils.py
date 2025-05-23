import logging
from typing import List
import hydra
from omegaconf import OmegaConf

from .types import ModelType
from ..utils import initialize_hydra


logger = logging.getLogger(__name__)


def list_available_models() -> List[str]:
    """
    Lists all available models defined in the models.yaml configuration file.

    Returns:
        list: A sorted list of model names available in the configuration.
    """
    initialize_hydra()

    # Load the datasets configuration
    cfg = OmegaConf.to_container(hydra.compose(config_name="models"), resolve=True)

    # Extract dataset names
    model_names = list(cfg.get("models", {}).keys())

    # Sort alphabetically for easier reading
    model_names.sort()

    return model_names


_MODEL_VARIANT_FINETUNE_TO_DISPLAY = {  # maps a tuple of (name, variant, fine_tune_dataset) to display (model name, details)
    # AIDO
    ("AIDO", "aido_cell_3m", None): ("AIDO", "Cell-3M"),
    # Geneformer
    ("GENEFORMER", "gf_6L_30M", None): ("GENEFORMER", "GF-6L-30M-i2048 (June 2021)"),
    ("GENEFORMER", "gf_12L_30M", None): ("GENEFORMER", "GF-12L-30M-i2048 (June 2021)"),
    ("GENEFORMER", "gf_12L_95M", None): ("GENEFORMER", "GF-12L-95M-i4096 (April 2024)"),
    ("GENEFORMER", "gf_20L_95M", None): ("GENEFORMER", "GF-20L-95M-i4096 (April 2024)"),
    # scGenePT
    (
        "SCGENEPT",
        "scgenept_ncbi_gpt",
        "adamson",
    ): ("scGenePT_{NCBI}", "fine-tuned, Adamson"),
    ("SCGENEPT", "scgenept_ncbi_gpt", "norman"): ("scGenePT_{NCBI}", "fine-tuned, Norman"),
    (
        "SCGENEPT",
        "scgenept_ncbi+uniprot_gpt",
        "adamson",
    ): ("scGenePT_{NCBI+UniProt}", "fine-tuned, Adamson"),
    (
        "SCGENEPT",
        "scgenept_ncbi+uniprot_gpt",
        "norman",
    ): ("scGenePT_{NCBI+UniProt}", "fine-tuned, Norman"),
    (
        "SCGENEPT",
        "scgenept_go_f_gpt_concat",
        "adamson",
    ): ("scGenePT_{GO-F}", "fine-tuned, Adamson"),
    (
        "SCGENEPT",
        "scgenept_go_f_gpt_concat",
        "norman",
    ): ("scGenePT_{GO-F}", "fine-tuned, Norman"),
    (
        "SCGENEPT",
        "scgenept_go_p_gpt_concat",
        "adamson",
    ): ("scGenePT_{GO-P}", "fine-tuned, Adamson"),
    (
        "SCGENEPT",
        "scgenept_go_p_gpt_concat",
        "norman",
    ): ("scGenePT_{GO-P}", "fine-tuned, Norman"),
    (
        "SCGENEPT",
        "scgenept_go_c_gpt_concat",
        "adamson",
    ): ("scGenePT_{GO-C}", "fine-tuned, Adamson"),
    (
        "SCGENEPT",
        "scgenept_go_c_gpt_concat",
        "norman",
    ): ("scGenePT_{GO-C}", "fine-tuned, Norman"),
    (
        "SCGENEPT",
        "scgenept_go_all_gpt_concat",
        "adamson",
    ): ("scGenePT_{GO-all}", "fine-tuned, Adamson"),
    (
        "SCGENEPT",
        "scgenept_go_all_gpt_concat",
        "norman",
    ): ("scGenePT_{GO-all}", "fine-tuned, Norman"),
    # scGPT
    ("SCGPT", "human", None): ("scGPT", "whole-human"),
    # scVI
    ("SCVI", "homo_sapiens", None): ("scVI", "Census 2023-12-15, homo sapiens"),
    # Transcriptformer
    ("TRANSCRIPTFORMER", "tf-sapiens", None): ("TRANSCRIPTFORMER", "TF-Sapiens"),
    ("TRANSCRIPTFORMER", "tf-exemplar", None): ("TRANSCRIPTFORMER", "TF-Exemplar"),
    ("TRANSCRIPTFORMER", "tf-metazoa", None): ("TRANSCRIPTFORMER", "TF-Metazoa"),
    # UCE
    ("UCE", "4l", None): ("UCE", "4L"),
    ("UCE", "33l", None): ("UCE", "33L"),
}


def model_to_display(
    model_type: ModelType, model_args: dict[str, str | int]
) -> tuple[str, str]:
    """try to map the model variant names to more uniform, pretty strings"""
    key = (model_type.name, model_args.get("model_variant"), model_args.get("dataset"))

    try:
        return _MODEL_VARIANT_FINETUNE_TO_DISPLAY[key]
    except KeyError:
        if model_args:
            logger.warning(
                "No display name provided for %r. Using title-case.", model_type
            )
        return (
            model_type.name.title(),
            ', '.join(f"{k}={v!r}" for k, v in model_args.items())
        )

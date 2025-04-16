def filter_adata_by_ensembl_ids(adata, ensembl_ids):
    """Filter AnnData object to include only genes with specified Ensembl IDs.

    Args:
        adata: AnnData object to filter
        ensembl_ids: List of Ensembl gene IDs to keep

    Returns:
        Filtered AnnData object
    """
    mask = adata.var["ensembl_id"].isin(ensembl_ids)
    return adata[:, mask].copy()


def prepare_adata_for_inference(adata):
    """Prepare AnnData object for TranscriptFormer inference.

    Args:
        adata: Input AnnData object

    Returns:
        Prepared AnnData object
    """
    # Ensure raw counts are available
    if "raw" not in adata.layers:
        adata.layers["raw"] = adata.X.copy()

    # Ensure ensembl_id column exists
    if "ensembl_id" not in adata.var.columns:
        raise ValueError("AnnData object must have 'ensembl_id' column in var")

    return adata

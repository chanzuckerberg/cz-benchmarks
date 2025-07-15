import numpy as np
import pandas as pd
from anndata import AnnData
from czbenchmarks.tasks.utils import align_adata_to_model_output


def test_align_adata_to_model_output_success():
    """Test successful alignment of AnnData and embeddings."""
    adata = AnnData(
        X=np.random.rand(5, 10),
        obs=pd.DataFrame(index=["cell1", "cell2", "cell3", "cell4", "cell5"]),
    )
    embeddings = np.random.rand(3, 3)  # Match model_output_index length
    model_output_index = ["cell3", "cell1", "cell5"]

    aligned_adata, aligned_embeddings = align_adata_to_model_output(
        adata, embeddings, model_output_index
    )

    assert list(aligned_adata.obs.index) == model_output_index
    assert aligned_embeddings.shape == (3, 3)


def test_align_adata_to_model_output_shape_mismatch():
    """Test alignment failure due to shape mismatch."""
    adata = AnnData(
        X=np.random.rand(5, 10),
        obs=pd.DataFrame(index=["cell1", "cell2", "cell3", "cell4", "cell5"]),
    )
    embeddings = np.random.rand(4, 3)  # Mismatched shape
    model_output_index = ["cell3", "cell1", "cell5"]

    try:
        align_adata_to_model_output(adata, embeddings, model_output_index)
    except ValueError as e:
        assert str(e) == (
            "Shape mismatch: The embeddings array has 4 rows, "
            "but the model_output_index has 3 entries."
        )


def test_align_adata_to_model_output_no_common_cells():
    """Test alignment failure due to no common cells."""
    adata = AnnData(
        X=np.random.rand(5, 10),
        obs=pd.DataFrame(index=["cell1", "cell2", "cell3", "cell4", "cell5"]),
    )
    embeddings = np.random.rand(3, 3)  # Match model_output_index length
    model_output_index = ["cell6", "cell7", "cell8"]  # No common cells

    try:
        align_adata_to_model_output(adata, embeddings, model_output_index)
    except ValueError as e:
        assert str(e) == (
            "Alignment failed: None of the cell IDs from the model output were found in the AnnData object."
        )


def test_align_adata_to_model_output_partial_overlap():
    """Test alignment when only some cells in model_output_index exist in adata."""
    adata = AnnData(
        X=np.random.rand(5, 10),
        obs=pd.DataFrame(index=["cell1", "cell2", "cell3", "cell4", "cell5"]),
    )
    embeddings = np.random.rand(3, 3)  # Match model_output_index length
    model_output_index = ["cell3", "cell1", "cell6"]  # Partial overlap

    aligned_adata, aligned_embeddings = align_adata_to_model_output(
        adata, embeddings, model_output_index
    )

    assert list(aligned_adata.obs.index) == ["cell3", "cell1"]
    assert aligned_embeddings.shape == (2, 3)

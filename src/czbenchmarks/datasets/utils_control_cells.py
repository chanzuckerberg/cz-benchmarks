import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import pairwise_distances
import logging

logger = logging.getLogger(__name__)


def histogram_subsample(A, B, bins='auto', subsample_size=None):
    """
    Subsample values in B to match the distribution of values in A.
    
    Parameters:
    - A: Reference distribution
    - B: Distribution to subsample
    - bins: Number of bins for histogram (default: 'auto')
    - subsample_size: Size of subsample (default: len(B))
    
    Returns:
    - Indices of selected elements from A
    """
    # Create shared bin edges
    combined = np.concatenate([A, B])
    bin_edges = np.histogram_bin_edges(combined, bins=bins)
    # Calculate target distribution from B
    hist_B, _ = np.histogram(B, bins=bin_edges)
    target_proportions = hist_B / hist_B.sum()
    # Assign weights to elements in A based on B's distribution
    bin_indices_A = np.clip(np.digitize(A, bin_edges) - 1, 0, len(bin_edges)-2)
    weights = target_proportions[bin_indices_A]
    # Normalize weights and subsample
    weights /= weights.sum()
    if subsample_size is None:
        subsample_size = len(B)
    selected_indices = np.random.choice(
            len(A) , 
            size=subsample_size, 
            replace=False, 
            p=weights
        )
    return selected_indices

def get_matched_controls(adata, perturbation, min_cells=50, matchtype='GEM', verbose=False, dict_ctrls=None, dataset = None, pert_column = None, ctrl_condition  = None, gem_column = None, libsize_column  = None):
    """
    Get matched control cells for a given perturbation in an experiment.
    
    Parameters:
    -----------
    adata : AnnData
        The AnnData object containing the data
    perturbation : str
        The perturbation to match controls for
    min_cells : int, optional
        Minimum number of perturbed cells required (default: 50)
    matchtype : str, optional
        Matching strategy to use:
        - 'GEM': Match based on GEM group and UMI counts
        - 'lib': Match based on UMI counts only, GEM-agnostic
    verbose : bool, optional
        Whether to print detailed statistics (default: False)
    dict_ctrls : dict, optional
        Dictionary mapping perturbations to control conditions
    dataset : str, optional
        Name of dataset to use predefined column names. Must be one of: ['replogle', 'adamson']
    pert_column : str, optional
        Name of column containing perturbation information. Required if dataset is None
    ctrl_condition : str, optional
        Value in pert_column that indicates control cells. Required if dataset is None
    gem_column : str, optional
        Name of column containing GEM group information. Required if dataset is None
    libsize_column : str, optional
        Name of column containing library size information. Required if dataset is None
    
    Returns:
    --------
    tuple
        (pert_cells, matched_controls) where:
        - pert_cells: List of perturbed cell names
        - matched_controls: List of matched control cell names
    Get matched control cells for a given perturbation in an experiment.
    
    """
    dataset_dict = {
        'replogle': ['gene', 'non-targeting', 'gem_group', 'UMI_count'],
        'adamson': ['perturbation_treatment', 'ctrl', 'gem', 'UMIcount']
    }

    if dataset is not None:
        if dataset in dataset_dict.keys():
            pert_column, ctrl_condition, gem_column, libsize_column = tuple(dataset_dict[dataset])
        else:
            raise ValueError(f"Dataset {dataset} not found. Must be one of: {list(dataset_dict.keys())}")
    # If dataset is None, check that all column fields are provided
    if dataset is None:
        if any(field is None for field in [pert_column, ctrl_condition, gem_column, libsize_column]):
            raise ValueError("When dataset is None, all column fields (pert_column, ctrl_condition, gem_column, libsize_column) must be provided")
        else:
            pert_column = pert_column
            ctrl_condition = ctrl_condition
            gem_column = gem_column
            libsize_column = libsize_column    

    # Get perturbed and control cells
    pert_cells = adata[adata.obs[pert_column] == perturbation]
    ctrl_cells = adata[adata.obs[pert_column] == ctrl_condition]
    
    if (dict_ctrls is not None) & (verbose):
        print(f"Using pre-computed matched controls for {perturbation}")
        return pert_cells.obs_names, dict_ctrls[perturbation]

    if len(pert_cells) < min_cells:
        print(f"Warning: Only {len(pert_cells)} cells for {perturbation}")
        return None, None
    
    if matchtype == 'GEM':
        # Convert to DataFrame for easier manipulation
        pert_df = pert_cells.obs.copy()
        ctrl_df = ctrl_cells.obs.copy()
        
        # Initialize matched controls list
        pert_df['matched_control'] = None
        
        # Track used controls for this perturbation
        used_controls = set()
        
        # Group by GEM and UMI count
        for (gem_group, umi_count), group in pert_df.groupby([gem_column, libsize_column]):
            # Get available controls in this GEM group
            available_ctrls = ctrl_df[ctrl_df[gem_column] == gem_group].copy()
            
            if len(available_ctrls) == 0:
                print(f"Warning: No controls available in GEM {gem_group}")
                continue
            
            # Calculate differences in UMI count
            available_ctrls['difference'] = np.abs(available_ctrls[libsize_column] - umi_count)
            
            # Remove controls that have already been used for this perturbation
            available_ctrls = available_ctrls[~available_ctrls.index.isin(used_controls)]
            
            if len(available_ctrls) == 0:
                print(f"Warning: No unused controls available in GEM {gem_group}")
                continue
            
            # Sort by difference and take the closest matches
            available_ctrls = available_ctrls.sort_values('difference')
            
            # Match controls to perturbed cells
            for i, (_, row) in enumerate(group.iterrows()):
                if i < len(available_ctrls):
                    matched_control = available_ctrls.index[i]
                    pert_df.loc[row.name, 'matched_control'] = matched_control
                    used_controls.add(matched_control)
       
        matched_controls = pert_df['matched_control'].dropna().to_dict()
    
    elif matchtype == 'lib':
        # Get library sizes for matching
        pert_libsize = pert_cells.obs[libsize_column].values
        ctrl_libsize = ctrl_cells.obs[libsize_column].values
        
        # Use histogram subsampling to match library size distributions
        matched_indices = histogram_subsample(
            ctrl_libsize,
            pert_libsize,
            bins='auto',
            subsample_size=len(pert_libsize)
        )
        
        matched_controls = ctrl_cells.obs_names[matched_indices].tolist()
    
    else:
        raise ValueError(f"Invalid matchtype: {matchtype}. Must be 'GEM' or 'lib'")
    
    # Print matching statistics
    if verbose:
        print(f"\nMatching statistics for {perturbation}:")
        print(f"Perturbed cells: {len(pert_cells)}")
        print(f"Matched controls: {len(matched_controls)}")
        
        # Calculate matching quality metrics
        if len(matched_controls) > 0:
            pert_sizes = pert_cells.obs[libsize_column]
            ctrl_sizes = ctrl_cells[matched_controls].obs[libsize_column]
            size_diff = np.abs(pert_sizes.values - ctrl_sizes.values)
            print(f"Median UMI count difference: {np.median(size_diff):.1f}")
            print(f"Mean UMI count difference: {np.mean(size_diff):.1f}")
            
            # Additional statistics for GEM matching
            if matchtype == 'GEM':
                gem_matches = sum(1 for ctrl in matched_controls 
                                if ctrl_cells.obs.loc[ctrl, gem_column] in pert_cells.obs[gem_column].unique())
                print(f"Controls matched within same GEM: {gem_matches}/{len(matched_controls)}")
    
    return pert_cells.obs_names, matched_controls


# TODO name is temporary
def get_matched_controls_v2(obs_metadata: pd.DataFrame, 
                           dist_keys: List[str], 
                           condition_key: str = 'condition', 
                           control_condition: str = 'non-targeting', 
                           group_key='gem_group') -> Dict[str, Dict[str, str]]:
    """
    Get matched control cells for a given perturbation in an experiment.
    Parameters:
    -----------
    obs_metadata: pd.DataFrame
        The metadata DataFrame containing the data
    dist_keys: List[str]
        The keys to use for the distance matrix
    condition_key: str
        The key to use for the condition
    control_condition: str
        The value to use for the control condition
    group_key: str
        The key to use for the group

    Returns:
    --------
    Dict[str, Dict[str, str]]
        A dictionary containing conditions as keys and then paired condition / control cell barcodes
    """
    # # Can probably accelerate by inverting loop order
    # # Will run fewer pairwise distance calculations, albeit with memory increase
    # for gem_group, treatment_gem_group in treatment_cells.groupby(group_key):
    #     control_ = control_cells_gem[gem_group][dist_keys]
    #     treatment_ = treatment_gem_group[dist_keys]
    #     dist_matrix = pairwise_distances(treatment_, control_)
        
    #     for condition, treatment_gem_condition in treatment_gem_group.groupby(condition_key):
    #         treatment_indices, control_indices = linear_sum_assignment(dist_matrix[treatment_gem_condition.indexes])

    #         treatment_ids = treatment_.index[treatment_indices]
    #         control_ids = control_.index[control_indices]
    #         control_cells_ids[condition].update(dict(zip(treatment_ids, control_ids)))

    # FIXME MICHELLE for debugging
    precision = 3
    np.set_printoptions(precision=precision)

    df_columns = [condition_key, group_key] + dist_keys
    obs_metadata = obs_metadata[df_columns]
    
    # Setup control and treatment cells
    control_mask = obs_metadata[condition_key] == control_condition
    control_cells = obs_metadata.loc[control_mask]
    treatment_cells = obs_metadata.loc[~control_mask]
    
    # Group control cells by gem group
    control_cells_gem = {gem_group:df for gem_group,df in control_cells.groupby(group_key)}


    control_cells_ids_filtered_lsum = defaultdict(dict)
    control_cells_ids_filtered_argm = defaultdict(dict)
    
    for condition, treatment_cells_cond in treatment_cells.groupby(condition_key):
        for gem_group, treatment_cells_cond_group in treatment_cells_cond.groupby(group_key):

            control_ = control_cells_gem[gem_group][dist_keys]
            treatment_ = treatment_cells_cond_group[dist_keys]
            
            if len(treatment_) > len(control_):
                logger.info(f"Warning: {condition}, {gem_group} num treatment cells exceeds num control cells. "
                f"{len(treatment_)} > {len(control_)}. This edge case has not been well tested."
                )
            elif len(treatment_) == len(control_):
                logger.info(f"Condition{condition}, gem group{gem_group} num treatment cells equals num control cells. "
                f"{len(treatment_)} = {len(control_)}"
                )
            
            dist_matrix = pairwise_distances(treatment_, control_)
            treatment_indices, control_indices = linear_sum_assignment(dist_matrix)
            
            if max(treatment_indices) >= len(treatment_): 
                raise AssertionError("Treatment indices out of range")
            if max(control_indices) >= len(control_): 
                raise AssertionError("Control indices out of range")

            control_indices_arg = dist_matrix.argmin(axis=1)
    
            # FIXME MICHELLE for debugging
            if len(control_indices) == len(control_indices_arg) and not np.allclose(
                control_indices[treatment_indices], control_indices_arg
            ):
                lsum_dist_values = dist_matrix[treatment_indices, control_indices]
                arg1_dist_values = dist_matrix[list(range(dist_matrix.shape[0])), control_indices_arg]
                logger.info(
                    f'Condition {condition}, gem_group {gem_group} :\n'
                    f'     lsum indices {control_indices[treatment_indices]} distances {lsum_dist_values} {sum(lsum_dist_values):.{precision}f}\n'
                    f'     arg1 indices {control_indices_arg} distances {arg1_dist_values} {sum(arg1_dist_values):.{precision}f}\n'
                )
            
            treatment_ids = treatment_.index[treatment_indices]
            control_ids = control_.index[control_indices]
            control_ids_arg = control_.index[control_indices_arg]
            control_cells_ids_filtered_lsum[condition].update(dict(zip(treatment_ids, control_ids)))
            # This is just for debugging
            control_cells_ids_filtered_argm[condition].update(dict(zip(treatment_.index, control_ids_arg)))
    
    return control_cells_ids_filtered_lsum
import numpy as np

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

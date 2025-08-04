from enum import Enum
from typing import Any, Callable, Dict, Optional, Set
from pydantic import BaseModel, Field


class MetricType(Enum):
    """Enumeration of all supported metric types.

    Defines unique identifiers for evaluation metrics that can be computed.
    Each metric type corresponds to a specific evaluation metric, and its value
    is used as a string identifier in results dictionaries.

    Examples:
        - Clustering metrics: Adjusted Rand Index, Normalized Mutual Information
        - Embedding quality metrics: Silhouette Score
        - Integration metrics: Entropy Per Cell, Batch Silhouette
        - Perturbation metrics: Mean Squared Error, Pearson Correlation
    """

    # Clustering metrics
    ADJUSTED_RAND_INDEX = "adjusted_rand_index"
    NORMALIZED_MUTUAL_INFO = "normalized_mutual_info"

    # Embedding quality metrics
    SILHOUETTE_SCORE = "silhouette_score"

    # Integration metrics
    ENTROPY_PER_CELL = "entropy_per_cell"
    BATCH_SILHOUETTE = "batch_silhouette"

    # Cross-validation prediction metrics
    MEAN_FOLD_ACCURACY = "mean_fold_accuracy"
    MEAN_FOLD_F1_SCORE = "mean_fold_f1"
    MEAN_FOLD_PRECISION = "mean_fold_precision"
    MEAN_FOLD_RECALL = "mean_fold_recall"
    MEAN_FOLD_AUROC = "mean_fold_auroc"

    MEAN_SQUARED_ERROR = "mean_squared_error"
    PEARSON_CORRELATION = "PEARSON_CORRELATION"
    JACCARD = "jaccard"
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    SPEARMAN_CORRELATION = "spearman_correlation"


class MetricInfo(BaseModel):
    """Stores metadata about a metric.

    Encapsulates information required for metric computation, including:
    - The function implementing the metric.
    - Required arguments for the metric function.
    - Default parameters for the metric function.
    - An optional description of the metric's purpose.
    - Tags for grouping related metrics.

    Attributes:
        func (Callable): The function that computes the metric.
        required_args (Set[str]): Names of required arguments for the metric function.
        default_params (Dict[str, Any]): Default parameters for the metric function.
        description (Optional[str]): Documentation string describing the metric.
        tags (Set[str]): Tags for categorizing metrics.
    """

    func: Callable
    """The function that computes the metric"""

    required_args: Set[str]
    """Set of required argument names"""

    default_params: Dict[str, Any]
    """Default parameters for the metric function"""

    description: Optional[str] = None
    """Optional documentation string for custom metrics"""

    tags: Set[str] = None
    """Set of tags for grouping related metrics"""


class MetricRegistry:
    """Central registry for all available metrics.

    Provides functionality for registering, validating, and computing metrics.
    Each metric is associated with a unique `MetricType` identifier and metadata
    stored in a `MetricInfo` object.

    Features:
    - Register new metrics with required arguments, default parameters, and tags.
    - Compute metrics by passing required arguments and merging with defaults.
    - Retrieve metadata about registered metrics.
    - List available metrics, optionally filtered by tags.

    Attributes:
        _metrics (Dict[MetricType, MetricInfo]): Internal storage for registered metrics.
    """

    def __init__(self):
        self._metrics: Dict[MetricType, MetricInfo] = {}

    def register(
        self,
        metric_type: MetricType,
        func: Callable,
        required_args: Optional[Set[str]] = None,
        default_params: Optional[Dict[str, Any]] = None,
        description: str = "",
        tags: Optional[Set[str]] = None,
    ) -> None:
        """Register a new metric in the registry.

        Associates a metric type with its computation function, required arguments,
        default parameters, and metadata. Registered metrics can later be computed
        using the `compute` method.

        Args:
            metric_type (MetricType): Unique identifier for the metric.
            func (Callable): Function that computes the metric.
            required_args (Optional[Set[str]]): Names of required arguments for the metric function.
            default_params (Optional[Dict[str, Any]]): Default parameters for the metric function.
            description (str): Documentation string describing the metric's purpose.
            tags (Optional[Set[str]]): Tags for categorizing the metric.

        Raises:
            TypeError: If `metric_type` is not an instance of `MetricType`.
        """

        if not isinstance(metric_type, MetricType):
            raise TypeError(
                f"Invalid metric type: {metric_type}. Must be a MetricType enum."
            )

        self._metrics[metric_type] = MetricInfo(
            func=func,
            required_args=required_args or set(),
            default_params=default_params or {},
            description=description,
            tags=tags or set(),
        )

    def compute(self, metric_type: MetricType, **kwargs) -> float:
        """Compute a registered metric with the given parameters.

        Validates required arguments and merges them with default parameters before
        calling the metric's computation function.

        Args:
            metric_type (MetricType): Type of metric to compute.
            **kwargs: Arguments to pass to the metric function.

        Returns:
            float: Computed metric value.

        Raises:
            ValueError: If the metric type is unknown or required arguments are missing.
        """
        if metric_type not in self._metrics:
            raise ValueError(f"Unknown metric type: {metric_type}")

        metric_info = self._metrics[metric_type]

        # Validate required arguments
        missing_args = metric_info.required_args - set(kwargs.keys())
        if missing_args:
            raise ValueError(
                f"Missing required arguments for {metric_type}: {missing_args}"
            )

        # Merge with defaults and compute
        params = {**metric_info.default_params, **kwargs}
        return metric_info.func(**params)

    def get_info(self, metric_type: MetricType) -> MetricInfo:
        """Get metadata about a metric.

        Args:
            metric_type: Type of metric

        Returns:
            MetricInfo object with metric metadata

        Raises:
            ValueError: If metric type unknown
        """
        if metric_type not in self._metrics:
            raise ValueError(f"Unknown metric type: {metric_type}")
        return self._metrics[metric_type]

    def list_metrics(self, tags: Optional[Set[str]] = None) -> Set[MetricType]:
        """List available metrics, optionally filtered by tags.

        Retrieves all registered metrics, or filters them based on the provided tags.

        Args:
            tags (Optional[Set[str]]): Tags to filter metrics. Only metrics with all
                specified tags will be returned.

        Returns:
            Set[MetricType]: Set of matching metric types.
        """
        if tags is None:
            return set(self._metrics.keys())

        return {
            metric_type
            for metric_type, info in self._metrics.items()
            if tags.issubset(info.tags)
        }


class MetricResult(BaseModel):
    """Represents the result of a single metric computation.

    Encapsulates the computed value, associated metric type, and any parameters
    used during computation. Provides functionality for generating aggregation keys
    to group similar metrics.

    Attributes:
        metric_type (MetricType): The type of metric computed.
        value (float): The computed metric value.
        params (Optional[Dict[str, Any]]): Parameters used during computation.

    Methods:
        aggregation_key: Generates a key based on the metric type and parameters
            to aggregate similar metrics together.
    """

    metric_type: MetricType
    value: float
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @property
    def aggregation_key(self) -> str:
        """return a key based on the metric type and params in order to aggregate the same metrics together"""
        if self.params is None:
            params = ""
        else:
            params = "_".join(
                (f"{key}-{value}" for key, value in sorted(self.params.items()))
            )
        return f"{self.metric_type}({params})"


class AggregatedMetricResult(BaseModel):
    """Represents the aggregated result of multiple metric computations.

    Stores statistical information about a set of metric values, including the
    mean, standard deviation, and raw values. Useful for summarizing metrics
    computed across multiple runs or folds.

    Attributes:
        metric_type (MetricType): The type of metric being aggregated.
        params (Dict[str, Any] | None): Parameters used during computation.
        n_values (int): Number of values aggregated.
        value (float): Mean value of the aggregated metrics.
        value_std_dev (float | None): Standard deviation of the aggregated metrics.
        values_raw (list[float]): Raw values of the metrics being aggregated.
    """

    metric_type: MetricType
    params: Dict[str, Any] | None = Field(default_factory=dict)
    n_values: int
    value: float
    value_std_dev: float | None
    values_raw: list[float]

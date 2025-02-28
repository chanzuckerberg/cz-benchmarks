from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set


class MetricType(Enum):
    """Enumeration of all supported metric types.

    Each metric type corresponds to a specific evaluation metric that can be computed.
    The value is the string identifier used in results dictionaries.
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

    # Regression metrics
    MSE = "mse"
    R2_SCORE = "r2_score"


@dataclass
class MetricInfo:
    """Stores metadata about a metric.

    Attributes:
        func: The function that computes the metric
        required_args: Set of required argument names
        default_params: Default parameters for the metric function
        description: Optional documentation string for custom metrics
        tags: Set of tags for grouping related metrics
    """

    func: Callable
    required_args: Set[str]
    default_params: Dict[str, Any]
    description: Optional[str] = None
    tags: Set[str] = None

    def __post_init__(self):
        """Validate fields after initialization."""
        if self.optimization_direction not in ("max", "min"):
            raise ValueError("optimization_direction must be 'max' or 'min'")
        if self.tags is None:
            self.tags = set()


class MetricRegistry:
    """Central registry for all available metrics.

    Handles registration and computation of metrics with proper validation.
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
        """Register a new metric.

        Args:
            metric_type: Type of metric to register
            func: Function that computes the metric
            required_args: Set of required argument names
            default_params: Default parameters for the metric function
            description: Documentation string
            tags: Set of tags for grouping metrics
        """

        self._metrics[metric_type] = MetricInfo(
            func=func,
            required_args=required_args or set(),
            default_params=default_params or {},
            description=description,
            tags=tags or set(),
        )

    def compute(self, metric_type: MetricType, **kwargs) -> float:
        """Compute a metric with the given parameters.

        Args:
            metric_type: Type of metric to compute
            **kwargs: Arguments to pass to metric function

        Returns:
            Computed metric value

        Raises:
            ValueError: If metric type unknown or missing required args
            ValueError: If computed value outside expected range
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

        Args:
            tags: If provided, only return metrics with all these tags

        Returns:
            Set of matching MetricType values
        """
        if tags is None:
            return set(self._metrics.keys())

        return {
            metric_type
            for metric_type, info in self._metrics.items()
            if tags.issubset(info.tags)
        }

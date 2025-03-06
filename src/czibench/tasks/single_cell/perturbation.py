from typing import Dict, Set
import pandas as pd
import logging
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
)
from ..base import BaseTask
from ...datasets.single_cell import PerturbationSingleCellDataset
from ...datasets.types import DataType
from ...metrics import MetricType

logger = logging.getLogger(__name__)


class PerturbationTask(BaseTask):
    """Task for evaluating perturbation prediction quality.

    This task computes metrics to assess how well a model predicts gene expression
    changes in response to perturbations. Compares predicted vs ground truth
    perturbation effects using MSE and correlation metrics.
    """

    @property
    def required_inputs(self) -> Set[DataType]:
        """Required input data types.

        Returns:
            Set of required input DataTypes (ground truth perturbation effects)
        """
        return {DataType.PERTURBATION_TRUTH}

    @property
    def required_outputs(self) -> Set[DataType]:
        """Required output data types.

        Returns:
            required output types from models this task to run
            (predicted perturbation effects)
        """
        return {DataType.PERTURBATION_PRED}

    def _run_task(self, data: PerturbationSingleCellDataset):
        """Runs the perturbation evaluation task.

        Gets predicted perturbation effects, ground truth effects, and control
        expression from the dataset for metric computation.

        Args:
            data: Dataset containing perturbation predictions and ground truth
        """
        self.perturbation_pred = data.get_output(DataType.PERTURBATION_PRED)
        self.perturbation_truth = data.perturbation_truth
        self.perturbation_ctrl = pd.Series(
            data=data.adata.X.mean(0).A.flatten(),
            index=data.adata.var_names,
            name="ctrl",
        )

    def _compute_metrics(self) -> Dict[MetricType, float]:
        """Computes perturbation prediction quality metrics.

        For each perturbation, computes:
        - MSE between predicted and true expression
        - Correlation between predicted and true expression changes from control

        Returns:
            Dictionary containing MSE and correlation metrics per perturbation
        """
        metrics = {}

        avg_perturbation_control = self.perturbation_ctrl

        MetricType.MEAN_SQUARED_ERROR
        MetricType.R2_SCORE

        for key in self.perturbation_pred.keys():
            if key in self.perturbation_truth.keys():
                metrics[key] = {}

                avg_perturbation_pred = self.perturbation_pred[key].mean(axis=0)
                avg_perturbation_truth = self.perturbation_truth[key].mean(axis=0)

                intersecting_genes = list(
                    set(avg_perturbation_pred.index)
                    & set(avg_perturbation_truth.index)
                    & set(avg_perturbation_control.index)
                )

                mse = mean_squared_error(
                    avg_perturbation_pred[intersecting_genes],
                    avg_perturbation_truth[intersecting_genes],
                )
                delta_pearson_corr = r2_score(
                    avg_perturbation_pred[intersecting_genes]
                    - avg_perturbation_control[intersecting_genes],
                    avg_perturbation_truth[intersecting_genes]
                    - avg_perturbation_control[intersecting_genes],
                )
                metrics[key]["mse"] = mse
                metrics[key]["delta_pearson_corr"] = delta_pearson_corr
            else:
                logger.warning(
                    f"Perturbation {key} is not available in the ground truth "
                    "test perturbations. Skipping metrics for this perturbation."
                )

        return metrics

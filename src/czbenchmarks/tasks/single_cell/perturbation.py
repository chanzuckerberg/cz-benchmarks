from typing import Dict, Set
import pandas as pd
import logging
from ..base import BaseTask
from ...datasets import PerturbationSingleCellDataset, DataType
from ...metrics import MetricType
from ...models.types import ModelType

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

    def _run_task(self, data: PerturbationSingleCellDataset, model_type: ModelType):
        """Runs the perturbation evaluation task.

        Gets predicted perturbation effects, ground truth effects, and control
        expression from the dataset for metric computation.

        Args:
            data: Dataset containing perturbation predictions and ground truth
        """
        self.gene_pert, self.perturbation_pred = data.get_output(
            model_type, DataType.PERTURBATION_PRED
        )
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

        mean_squared_error_metric = MetricType.MEAN_SQUARED_ERROR
        r2_score_metric = MetricType.R2_SCORE

        if self.gene_pert in self.perturbation_truth.keys():

            avg_perturbation_pred = self.perturbation_pred.mean(axis=0)
            avg_perturbation_truth = self.perturbation_truth[self.gene_pert].mean(
                axis=0
            )

            intersecting_genes = list(
                set(avg_perturbation_pred.index)
                & set(avg_perturbation_truth.index)
                & set(avg_perturbation_control.index)
            )

            mse = metrics.compute(
                mean_squared_error_metric,
                y_true=avg_perturbation_truth[intersecting_genes],
                y_pred=avg_perturbation_pred[intersecting_genes],
            )
            delta_pearson_corr = metrics.compute(
                r2_score_metric,
                y_true=avg_perturbation_truth[intersecting_genes]
                - avg_perturbation_control[intersecting_genes],
                y_pred=avg_perturbation_pred[intersecting_genes]
                - avg_perturbation_control[intersecting_genes],
            )

            return {
                mean_squared_error_metric.value: mse,
                r2_score_metric.value: delta_pearson_corr,
            }
        else:
            raise ValueError(
                f"Perturbation {self.gene_pert} is not available in the ground truth "
                "test perturbations."
            )

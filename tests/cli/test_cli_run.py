import argparse
from pathlib import Path
from unittest.mock import MagicMock, call

from pytest_mock import MockFixture

from czbenchmarks import runner
from czbenchmarks.cli.cli_run import (
    get_model_arg_permutations,
    get_processed_dataset_cache_path,
    main,
    ModelArgs,
    ModelArgsDict,
    run_task,
    run_with_inference,
    run_without_inference,
    set_processed_datasets_cache,
    TaskArgs,
    TaskResult,
)
from czbenchmarks.constants import PROCESSED_DATASETS_CACHE_PATH
from czbenchmarks.datasets import utils as dataset_utils
from czbenchmarks.metrics.types import MetricResult, MetricType
from czbenchmarks.models.types import ModelType
from czbenchmarks.tasks.clustering import ClusteringTask
from czbenchmarks.tasks.embedding import EmbeddingTask


def test_main(mocker: MockFixture) -> None:
    # Setup mocks
    mock_task_results = []
    mock_run = mocker.patch(
        "czbenchmarks.cli.cli_run.run", return_value=mock_task_results
    )
    mock_write_results = mocker.patch(
        "czbenchmarks.cli.cli_run.write_results",
        return_value=None,
    )
    mock_task_args = MagicMock()
    mock_parse_task_args = mocker.patch(
        "czbenchmarks.cli.cli_run.parse_task_args", return_value=mock_task_args
    )

    # Handle empty inputs
    main(
        argparse.Namespace(
            models=[],
            tasks=[],
            datasets=[],
            output_file=None,
            output_format=None,
            batch_json=[],
        )
    )
    mock_run.assert_called_once_with(dataset_names=[], model_args=[], task_args=[])
    mock_write_results.assert_called_once_with(
        mock_task_results, output_format=None, output_file=None
    )
    mock_parse_task_args.assert_not_called()

    # Reset mocked functions
    mock_parse_task_args.reset_mock()
    mock_run.reset_mock()
    mock_write_results.reset_mock()

    # Handle complex inputs
    main(
        argparse.Namespace(
            models=["SCGPT", "SCVI"],
            tasks=["embedding", "clustering"],
            embedding_task_label_key=["cell_type"],
            clustering_task_label_key=["cell_type"],
            clustering_task_set_baseline=True,
            scgpt_model_variant=["human"],
            scvi_model_variant=["homo_sapiens", "mus_musculus"],
            datasets=["tsv2_blood", "tsv2_heart"],
            output_file="output_file.yaml",
            output_format="yaml",
            batch_json=[""],
        )
    )
    mock_run.assert_called_once_with(
        dataset_names=["tsv2_blood", "tsv2_heart"],
        model_args=[
            ModelArgs(name="SCGPT", args={"model_variant": ["human"]}),
            ModelArgs(
                name="SCVI", args={"model_variant": ["homo_sapiens", "mus_musculus"]}
            ),
        ],
        task_args=[mock_task_args, mock_task_args],
    )
    mock_write_results.assert_called_once_with(
        mock_task_results, output_format="yaml", output_file="output_file.yaml"
    )
    assert mock_parse_task_args.call_count == 2

    # Reset mocked functions
    mock_parse_task_args.reset_mock()
    mock_run.reset_mock()
    mock_write_results.reset_mock()

    # Handle batch inputs
    main(
        argparse.Namespace(
            models=["SCGENEPT"],
            tasks=["perturbation"],
            datasets=[],
            output_file="output_file.yaml",
            output_format="yaml",
            batch_json=[
                '{"datasets": ["adamson_perturb"], "scgenept_dataset_name": ["adamson"], "scgenept_gene_pert": ["AEBPB+ctrl", "AEBPB+dox"]}',
                '{"datasets": ["norman_perturb"], "scgenept_dataset_name": ["norman"], "scgenept_gene_pert": ["NTGC+ctrl", "NTGC+dox"]}',
            ],
        )
    )
    mock_run.assert_has_calls(
        [
            call(
                dataset_names=["adamson_perturb"],
                model_args=[
                    ModelArgs(
                        name="SCGENEPT",
                        args={
                            "dataset_name": ["adamson"],
                            "gene_pert": ["AEBPB+ctrl", "AEBPB+dox"],
                        },
                    )
                ],
                task_args=[mock_task_args],
            ),
            call(
                dataset_names=["norman_perturb"],
                model_args=[
                    ModelArgs(
                        name="SCGENEPT",
                        args={
                            "dataset_name": ["norman"],
                            "gene_pert": ["NTGC+ctrl", "NTGC+dox"],
                        },
                    )
                ],
                task_args=[mock_task_args],
            ),
        ]
    )


def test_run_with_inference(mocker: MockFixture) -> None:
    # Setup mocks
    mock_processed_data = MagicMock()
    mock_load_dataset = mocker.patch.object(
        dataset_utils, "load_dataset", return_value=mock_processed_data
    )
    mock_run_inference = mocker.patch.object(
        runner, "run_inference", return_value=mock_processed_data
    )
    mock_task_results = [MagicMock()]
    mock_run_task = mocker.patch(
        "czbenchmarks.cli.cli_run.run_task", return_value=mock_task_results
    )
    dataset_names = ["tsv2_blood", "tsv2_heart"]
    model_args = [
        ModelArgs(name="SCGPT", args={}),
        ModelArgs(
            name="SCVI",
            args={"model_variant": ["homo_sapiens", "mus_musculus"]},
        ),
    ]
    embedding_task_args = TaskArgs(
        name="embedding",
        task=EmbeddingTask(label_key="cell_type"),
        set_baseline=False,
    )
    clustering_task_args = TaskArgs(
        name="clustering",
        task=ClusteringTask(label_key="cell_type"),
        set_baseline=True,
    )
    task_args = [embedding_task_args, clustering_task_args]

    # Run tasks with mocked data
    task_results = run_with_inference(
        dataset_names=dataset_names,
        model_args=model_args,
        task_args=task_args,
    )

    # Verify results
    assert mock_load_dataset.call_count == 6  # 2 datasets * 3 model variants
    assert len(task_results) == 12  # # 2 datasets * 3 model variants * 2 tasks

    # Check that inference was run for each model variant, for each dataset
    assert mock_run_inference.call_args_list == [
        call("SCGPT", mock_processed_data, gpu=True),
        call("SCVI", mock_processed_data, gpu=True, model_variant="homo_sapiens"),
        call("SCVI", mock_processed_data, gpu=True, model_variant="mus_musculus"),
        call("SCGPT", mock_processed_data, gpu=True),
        call("SCVI", mock_processed_data, gpu=True, model_variant="homo_sapiens"),
        call("SCVI", mock_processed_data, gpu=True, model_variant="mus_musculus"),
    ]

    # Check that each task was run for each model variant, for each dataset
    assert mock_run_task.call_args_list == [
        call(
            "tsv2_blood",
            mock_processed_data,
            {"SCGPT": {}},
            embedding_task_args,
        ),
        call(
            "tsv2_blood",
            mock_processed_data,
            {"SCGPT": {}},
            clustering_task_args,
        ),
        call(
            "tsv2_blood",
            mock_processed_data,
            {"SCVI": {"model_variant": "homo_sapiens"}},
            embedding_task_args,
        ),
        call(
            "tsv2_blood",
            mock_processed_data,
            {"SCVI": {"model_variant": "homo_sapiens"}},
            clustering_task_args,
        ),
        call(
            "tsv2_blood",
            mock_processed_data,
            {"SCVI": {"model_variant": "mus_musculus"}},
            embedding_task_args,
        ),
        call(
            "tsv2_blood",
            mock_processed_data,
            {"SCVI": {"model_variant": "mus_musculus"}},
            clustering_task_args,
        ),
        call(
            "tsv2_heart",
            mock_processed_data,
            {"SCGPT": {}},
            embedding_task_args,
        ),
        call(
            "tsv2_heart",
            mock_processed_data,
            {"SCGPT": {}},
            clustering_task_args,
        ),
        call(
            "tsv2_heart",
            mock_processed_data,
            {"SCVI": {"model_variant": "homo_sapiens"}},
            embedding_task_args,
        ),
        call(
            "tsv2_heart",
            mock_processed_data,
            {"SCVI": {"model_variant": "homo_sapiens"}},
            clustering_task_args,
        ),
        call(
            "tsv2_heart",
            mock_processed_data,
            {"SCVI": {"model_variant": "mus_musculus"}},
            embedding_task_args,
        ),
        call(
            "tsv2_heart",
            mock_processed_data,
            {"SCVI": {"model_variant": "mus_musculus"}},
            clustering_task_args,
        ),
    ]


def test_run_without_inference(mocker: MockFixture) -> None:
    # Setup mocks
    mock_processed_data = MagicMock()
    mock_load_dataset = mocker.patch.object(
        dataset_utils, "load_dataset", return_value=mock_processed_data
    )
    mock_task_results = [MagicMock()]
    mock_run_task = mocker.patch(
        "czbenchmarks.cli.cli_run.run_task", return_value=mock_task_results
    )
    dataset_names = ["tsv2_blood", "tsv2_heart"]
    embedding_task_args = TaskArgs(
        name="embedding",
        task=EmbeddingTask(label_key="cell_type"),
        set_baseline=False,
    )
    clustering_task_args = TaskArgs(
        name="clustering",
        task=ClusteringTask(label_key="cell_type"),
        set_baseline=True,
    )
    task_args = [embedding_task_args, clustering_task_args]

    # Run tasks with mocked data
    task_results = run_without_inference(
        dataset_names=dataset_names,
        task_args=task_args,
    )

    # Verify results
    assert mock_load_dataset.call_count == 2  # once for each dataset
    assert len(task_results) == 4  # # 2 datasets * 2 tasks

    # Check that each task was run for each dataset
    assert mock_run_task.call_args_list == [
        call(
            "tsv2_blood",
            mock_processed_data,
            {},
            embedding_task_args,
        ),
        call(
            "tsv2_blood",
            mock_processed_data,
            {},
            clustering_task_args,
        ),
        call(
            "tsv2_heart",
            mock_processed_data,
            {},
            embedding_task_args,
        ),
        call(
            "tsv2_heart",
            mock_processed_data,
            {},
            clustering_task_args,
        ),
    ]


def test_run_task() -> None:
    # Setup mocks
    mock_dataset = MagicMock()
    model_args: dict[str, ModelArgsDict] = {
        ModelType.SCVI.value: {"model_variant": "homo_sapiens"}
    }
    mock_task_args = MagicMock()
    mock_task_args.name = "clustering"
    mock_task_run_result = {
        ModelType.SCVI: [
            MetricResult(
                metric_type=MetricType.ADJUSTED_RAND_INDEX, value=0.1, params={}
            )
        ]
    }
    mock_task_args.task.run.return_value = mock_task_run_result

    # Run task and check results
    task_results = run_task(
        "tsv2_heart",
        dataset=mock_dataset,
        model_args=model_args,
        task_args=mock_task_args,
    )
    assert task_results == [
        TaskResult(
            task_name="clustering",
            model_type="SCVI",
            dataset_name="tsv2_heart",
            model_args={"model_variant": "homo_sapiens"},
            metrics=[
                MetricResult(
                    metric_type=MetricType.ADJUSTED_RAND_INDEX, value=0.1, params={}
                )
            ],
        )
    ]


def test_get_model_arg_permutations(mocker: MockFixture) -> None:
    # 0 permutations for empty input
    assert get_model_arg_permutations([]) == {}

    # 1 (empty) permutation for a model with no args
    assert get_model_arg_permutations([ModelArgs(name="SCGENEPT", args={})]) == {
        "SCGENEPT": [{}]
    }

    # 1 permutation for a model with a single arg
    assert get_model_arg_permutations(
        [ModelArgs(name="SCGENEPT", args={"model_variant": ["norman"]})]
    ) == {"SCGENEPT": [{"model_variant": "norman"}]}

    # 2 permutations for a model with 1 set of  2 args
    assert get_model_arg_permutations(
        [ModelArgs(name="SCGENEPT", args={"model_variant": ["norman", "adamson"]})]
    ) == {
        "SCGENEPT": [{"model_variant": "norman"}, {"model_variant": "adamson"}],
    }

    # 4 permutations for a model with 2 sets of 2 args
    assert get_model_arg_permutations(
        [
            ModelArgs(
                name="SCGENEPT",
                args={
                    "model_variant": ["norman", "adamson"],
                    "gene_pert": ["CEBPB+ctrl", "CEBPB+dox"],
                },
            )
        ]
    ) == {
        "SCGENEPT": [
            {"model_variant": "norman", "gene_pert": "CEBPB+ctrl"},
            {"model_variant": "norman", "gene_pert": "CEBPB+dox"},
            {"model_variant": "adamson", "gene_pert": "CEBPB+ctrl"},
            {"model_variant": "adamson", "gene_pert": "CEBPB+dox"},
        ]
    }


def test_get_processed_dataset_cache_path() -> None:
    # The cache key for a model with no args is {dataset_name}_{model_name}.dill
    assert (
        get_processed_dataset_cache_path("tsv2_heart", model_name="SCVI", model_args={})
        == Path("~/.cz-benchmarks/processed_datasets/tsv2_heart_SCVI.dill")
        .expanduser()
        .absolute()
    )
    # Model args are sorted and included in the cache key
    assert (
        get_processed_dataset_cache_path(
            "norman_perturb",
            model_name="SCGENEPT",
            model_args={"model_variant": "norman", "gene_pert": "CEBPB+ctrl"},
        )
        == Path(
            "~/.cz-benchmarks/processed_datasets/norman_perturb_SCGENEPT_gene_pert-CEBPB+ctrl_model_variant-norman.dill"
        )
        .expanduser()
        .absolute()
    )


def test_set_processed_datasets_cache() -> None:
    mock_dataset = MagicMock()
    set_processed_datasets_cache(
        dataset=mock_dataset,
        dataset_name="tsv2_heart",
        model_name="SCVI",
        model_args={"model_variant": "homo_sapiens"},
    )
    mock_dataset.unload_data.assert_called_once()
    mock_dataset.serialize.assert_called_once_with(
        str(
            (
                Path(PROCESSED_DATASETS_CACHE_PATH)
                / "tsv2_heart_SCVI_model_variant-homo_sapiens.dill"
            )
            .expanduser()
            .absolute()
        )
    )
    mock_dataset.load_data.assert_called_once()

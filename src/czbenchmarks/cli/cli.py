import typer
import yaml
import json
import itertools
import glob
import numpy as np
import pandas as pd
import importlib.metadata
import sys
from pathlib import Path
from datetime import datetime, timezone
from czbenchmarks.datasets.utils import list_available_datasets, load_dataset
from czbenchmarks.tasks.clustering import ClusteringTask, ClusteringTaskInput
from czbenchmarks.tasks.embedding import EmbeddingTask, EmbeddingTaskInput
from czbenchmarks.tasks.label_prediction import (
    MetadataLabelPredictionTask,
    MetadataLabelPredictionTaskInput,
)
from czbenchmarks.tasks.integration import (
    BatchIntegrationTask,
    BatchIntegrationTaskInput,
)
from czbenchmarks.tasks.single_cell.cross_species import (
    CrossSpeciesIntegrationTask,
    CrossSpeciesIntegrationTaskInput,
)
from czbenchmarks.tasks.single_cell.perturbation import (
    PerturbationTask,
    PerturbationTaskInput,
)

app = typer.Typer(
    add_completion=False,
    help=(
        "CZ-Benchmarks CLI: A command-line utility for running benchmark tasks\n"
        "\n"
        "Example usage:\n"
        "\n"
        "-  List datasets: czbenchmarks list datasets\n"
        "\n"
        "-  List tasks: czbenchmarks list tasks\n"
        "\n"
        "-  Run tasks: czbenchmarks run --datasets dataset1 --embeddings embedding_pattern --tasks clustering\n"
        "\n"
        "-  Run with matrix: czbenchmarks run --matrix path/to/matrix.yaml\n"
    ),
)

try:
    _ver = importlib.metadata.version("cz-benchmarks")
except importlib.metadata.PackageNotFoundError:
    _ver = "unknown"
VERSION = f"v{_ver}"


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-v", help="Show CLI version and exit"
    ),
):
    """
    CZ-Benchmarks CLI: A command-line utility for running benchmark tasks
    Example usage:
    - List datasets: `czbenchmarks list datasets`
    - List tasks: `czbenchmarks list tasks`
    - Run tasks: `czbenchmarks run --datasets dataset1 --embeddings embedding_pattern --tasks clustering`
    - Run with matrix: `czbenchmarks run --matrix path/to/matrix.yaml`
    """
    if version:
        typer.echo(VERSION)
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


TASK_MAP = {
    "clustering": (ClusteringTask, ClusteringTaskInput),
    "embedding": (EmbeddingTask, EmbeddingTaskInput),
    "label_prediction": (MetadataLabelPredictionTask, MetadataLabelPredictionTaskInput),
    "integration": (BatchIntegrationTask, BatchIntegrationTaskInput),
    "cross_species": (CrossSpeciesIntegrationTask, CrossSpeciesIntegrationTaskInput),
    "perturbation": (PerturbationTask, PerturbationTaskInput),
}


@app.command("list")
def list_resources(
    resource: str = typer.Argument(
        None, help="Resource type to list: 'datasets' or 'tasks'", show_default=False
    ),
    datasets: bool = typer.Option(
        False, "--datasets", "-d", help="List available datasets"
    ),
    tasks: bool = typer.Option(False, "--tasks", "-t", help="List available tasks"),
):
    """
    List available datasets or tasks.
    Example:
    czbenchmarks list datasets
    czbenchmarks list tasks
    """
    if datasets:
        for ds in list_available_datasets():
            typer.echo(ds)
        return
    if tasks:
        for t in sorted(TASK_MAP.keys()):
            typer.echo(t)
        return
    if resource is None:
        typer.echo(
            "Specify resource to list: datasets or tasks\n"
            "Usage:\n"
            "  czbenchmarks list datasets   \n"
            "  czbenchmarks list tasks      \n"
            "Or use flags:\n"
            "  czbenchmarks list --datasets\n"
            "  czbenchmarks list --tasks"
        )
        raise typer.Exit(code=1)
    if resource == "datasets":
        for ds in list_available_datasets():
            typer.echo(ds)
    elif resource == "tasks":
        for t in sorted(TASK_MAP.keys()):
            typer.echo(t)
    else:
        typer.echo("Invalid argument. Use 'datasets' or 'tasks'.")
        raise typer.Exit(code=1)


@app.command("run")
def run(
    datasets: list[str] = typer.Option(..., "-d", "--datasets", help="Dataset names."),
    embeddings: list[str] = typer.Option(
        [],
        "-e",
        "--embeddings",
        help="Embedding file globs (supports {dataset},{seed}).",
    ),
    tasks: list[str] = typer.Option(..., "-t", "--tasks", help="Tasks to execute."),
    label_key: str = typer.Option(None, "--label-key", help="obs column for labels."),
    batch_key: str = typer.Option(
        None, "--batch-key", help="obs column for batch labels."
    ),
    clustering_iters: int = typer.Option(
        10, "--clustering-iters", help="Leiden iterations."
    ),
    clustering_flavor: str = typer.Option(
        "igraph", "--clustering-flavor", help="Leiden flavor."
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed override."),
    n_folds: int = typer.Option(5, "--n-folds", help="Folds for cross-validation."),
    min_class_size: int = typer.Option(
        10, "--min-class-size", help="Min class size for label prediction."
    ),
    perturbation_gene: str = typer.Option(
        None, "--perturbation-gene", help="Gene perturbation to evaluate."
    ),
    with_baseline: bool = typer.Option(
        False, "--with-baseline", help="Also run each task on its baseline."
    ),
    batch_json: list[Path] = typer.Option(
        [], "--batch-json", help="JSON files or strings for batch overrides."
    ),
    batch_seeds: list[int] = typer.Option(
        [], "--batch-random-seeds", help="List of random seeds for batching."
    ),
    matrix: Path = typer.Option(
        None, "-m", "--matrix", exists=True, help="YAML matrix for grid batches."
    ),
    output_format: str = typer.Option("json", "--output-format", help="json or yaml."),
    output_file: Path = typer.Option(
        None, "--output-file", help="Output file (stdout if omitted)."
    ),
    ctx: typer.Context = typer.Argument(None),
):
    """
    Run benchmarking tasks using precomputed embeddings or baselines. Supports free-form JSON batches, seed batches, or matrix YAML.

    Example:\n\n
            czbenchmarks run --datasets dataset1 --embeddings "embeddings/{dataset}_*.npy" --tasks clustering --label-key cell_type
    """

    combos = [
        {
            "seed": seed,
            "n_folds": n_folds,
            "min_class_size": min_class_size,
        }
    ]

    json_batches = []
    for bj in batch_json:
        raw = bj.read_text() if bj.is_file() else bj.read_text()
        parsed = yaml.safe_load(raw)
        if isinstance(parsed, list):
            json_batches.extend(parsed)
        elif isinstance(parsed, dict):
            json_batches.append(parsed)
    if not json_batches:
        json_batches = [{}]

    seed_batches = [{"seed": s} for s in batch_seeds] or [{}]

    combos = [dict(**j, **s) for j in json_batches for s in seed_batches]

    if matrix:
        mat = yaml.safe_load(matrix.read_text())
        ds_list = mat.get("datasets", datasets)
        emb_list = mat.get("embeddings", embeddings)
        task_list = mat.get("tasks", tasks)
        grid = mat.get("matrix", {})
        keys, vals = zip(*grid.items()) if grid else ([], [])
        grid_combos = (
            [dict(zip(keys, combo)) for combo in itertools.product(*vals)]
            if keys
            else [{}]
        )
        datasets, embeddings, tasks = ds_list, emb_list, task_list
        combos = [{**base, **g} for base in combos for g in grid_combos]

    if not datasets or not tasks:
        typer.echo("`--datasets` and `--tasks` required.")
        raise typer.Exit(code=1)

    all_results = []
    for combo in combos:
        single = [t for t in tasks if not TASK_MAP[t][0]().requires_multiple_datasets]
        multi = [t for t in tasks if TASK_MAP[t][0]().requires_multiple_datasets]

        ds_objs = {ds: load_dataset(ds) for ds in datasets}

        for ds in datasets:
            ds_obj = ds_objs[ds]
            raw = (
                ds_obj.adata.X.toarray()
                if hasattr(ds_obj.adata.X, "toarray")
                else ds_obj.adata.X
            )

            emb_files = []
            if embeddings:
                pats = [pat.format(dataset=ds, **combo) for pat in embeddings]
                emb_files = sum([glob.glob(p) for p in pats], [])
            if not emb_files and not single == ["perturbation"]:
                typer.secho(f"No embeddings for {ds}: {pats}", fg=typer.colors.RED)
                continue

            for tname in single:
                TaskCls, InputCls = TASK_MAP[tname]
                task = TaskCls(random_seed=combo.get("seed"))

                def run_one(data, tag, ti):
                    for mr in task.run(data, ti):
                        all_results.append(
                            {
                                "dataset": ds,
                                "task": tname,
                                "mode": tag,
                                "seed": combo.get("seed"),
                                "metric": mr.metric_type.value,
                                "value": mr.value,
                                **({"params": mr.params} if mr.params else {}),
                            }
                        )

                if tname == "perturbation":
                    genes = list(ds_obj.adata.var_names)
                    pred = pd.DataFrame(np.load(emb_files[0]), columns=genes)
                    ti = InputCls(
                        var_names=genes,
                        gene_pert=perturbation_gene
                        or list(ds_obj.perturbation_truth.keys())[0],
                        perturbation_pred=pred,
                        perturbation_truth=ds_obj.perturbation_truth,
                    )
                    run_one(raw, "model", ti)
                    if with_baseline:
                        pass
                else:
                    emb = np.load(emb_files[0])

                    if tname == "clustering":
                        ti = InputCls(
                            obs=ds_obj.adata.obs,
                            input_labels=ds_obj.adata.obs[label_key],
                            use_rep="X",
                            n_iterations=combo.get(
                                "clustering_iters", clustering_iters
                            ),
                            flavor=combo.get("clustering_flavor", clustering_flavor),
                        )
                    elif tname == "embedding":
                        ti = InputCls(input_labels=ds_obj.adata.obs[label_key])
                    elif tname == "label_prediction":
                        ti = InputCls(
                            labels=ds_obj.adata.obs[label_key],
                            n_folds=combo.get("n_folds", n_folds),
                            min_class_size=combo.get("min_class_size", min_class_size),
                        )
                    elif tname == "integration":
                        ti = InputCls(
                            labels=ds_obj.adata.obs[label_key],
                            batch_labels=ds_obj.adata.obs[batch_key],
                        )
                    run_one(emb, "model", ti)
                    if with_baseline and hasattr(task, "compute_baseline"):
                        base = task.compute_baseline(raw)
                        run_one(base, "baseline", ti)

        if multi:
            raws = []
            embs = []
            for ds in datasets:
                ds_obj = ds_objs[ds]
                raws.append(ds_obj.adata.X.toarray())
                pats = [pat.format(dataset=ds, **combo) for pat in embeddings]
                files = sum([glob.glob(p) for p in pats], [])
                if len(files) != 1:
                    raise typer.Exit(f"Need 1 emb per ds for {ds}, got {files}")
                embs.append(np.load(files[0]))

            for tname in multi:
                TaskCls, InputCls = TASK_MAP[tname]
                task = TaskCls(random_seed=combo.get("seed"))
                if tname == "cross_species":
                    labels_list = [obj.adata.obs[label_key] for obj in ds_objs.values()]
                    orgs = [obj.organism for obj in ds_objs.values()]
                    ti = InputCls(labels=labels_list, organism_list=orgs)
                    for mr in task.run(embs, ti):
                        all_results.append(
                            {
                                "datasets": ",".join(datasets),
                                "task": tname,
                                "mode": "model",
                                "seed": combo.get("seed"),
                                "metric": mr.metric_type.value,
                                "value": mr.value,
                                **({"params": mr.params} if mr.params else {}),
                            }
                        )

    out = {
        "czbenchmarks_version": VERSION,
        "args": "czbenchmarks " + " ".join(sys.argv[1:]),
        "task_results": all_results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if output_format.lower() == "yaml":
        text = yaml.dump(out)
    else:
        text = json.dumps(out, indent=2, default=str)
    if output_file:
        output_file.write_text(text)
        typer.echo(f"Wrote results to {output_file}")
    else:
        typer.echo(text)


def main():
    """
    Main entry point for the CLI.
    """
    app()


if __name__ == "__main__":
    app()

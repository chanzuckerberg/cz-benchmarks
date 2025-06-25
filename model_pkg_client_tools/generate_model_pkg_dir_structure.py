#!/usr/bin/env python3
"""
generate_model_pkg_dir_structure.py
===================================

CLI tool to scaffold an MLflow packaging project using a config-driven archetype.

This script performs the following operations:

1. **Copy** a package template (`pkg_templates/<template>`) into a new
   `<model_name>_mlflow_pkg` directory under your `--target`.
2. **Render** the Jinja stub for your chosen archetype:
   `archetypes/<archetype>_model.py.jinja` →
   `model_code/<model_name>_mlflow_model.py` in the target.
3. **Copy** the static spec stub:
   `archetypes/<archetype>_model_spec.py` → `model_spec.py` in the target.
4. **Clean up** by removing all `.jinja` templates from the target.
5. **Remove** the now-empty `model_code/archetypes/` directory in the target.
6. **Substitute** tokens `{{MODEL_NAME}}` and `{{MODEL_CAMEL}}` in all text files.

No user code is executed during scaffolding; this only creates a skeleton for
implementation.

CLI SWITCHES
------------
--model-name                 (required) Snake_case name for your model package.
--target                     (required) Existing directory under which
                              `<model_name>_mlflow_pkg` will be created.
--pkg-template               Subfolder in `pkg_templates/` to copy
                              (default: "mlflow_template").
--model-archetype            Archetype slug to render
                              (e.g. "file_uri_to_tensor").
--class-suffix               Suffix appended to the generated class
                              name (default: "MLflowModel").
--force-template-overwrite   If destination exists, back it up and overwrite.
--dry-run                    Print planned actions without writing files.

Examples
--------
1) Scaffold a new file-URI model package:
   python generate_model_pkg_dir_structure.py \
     --model-name cool_embedding \
     --target ~/projects \
     --pkg-template mlflow_template \
     --model-archetype file_uri_to_tensor

2) Scaffold and overwrite an existing folder:
   python generate_model_pkg_dir_structure.py \
     --model-name sample_model \
     --target ./workdir \
     --force-template-overwrite
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import shutil
import sys
from pathlib import Path
from typing import Iterable

import jinja2  # templating engine

_LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    """
    Ensure a root logger handler is configured exactly once.
    No-op if logging is already set up.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _backup_existing(dest: Path) -> None:
    """
    Rename an existing destination directory to a timestamped backup.

    Parameters
    ----------
    dest : Path
        The directory to back up.
    """
    ts = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    backup = dest.with_name(f"{dest.name}_backup_{ts}")
    dest.rename(backup)
    _LOGGER.warning("Destination exists – backed up to %s", backup)


def _iter_text_files(paths: Iterable[Path]) -> Iterable[Path]:
    """
    Yield probable text files for token substitution.

    Parameters
    ----------
    paths : Iterable[Path]
        Collection of file paths to filter by extension.

    Returns
    -------
    Iterable[Path]
    """
    exts = {".py", ".md", ".txt", ".rst", ".cfg", ".yml", ".yaml"}
    for p in paths:
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _ignore_docs(dirpath: str, names: list[str]) -> set[str]:
    """
    Instruct copytree to skip only PACKAGING_INSTRUCTIONS.md.

    Parameters
    ----------
    dirpath : str
        Directory being copied.
    names : list[str]
        Names in that directory.

    Returns
    -------
    set[str]
        Filenames to ignore.
    """
    skip = {"PACKAGING_INSTRUCTIONS.md"}
    return {n for n in names if n in skip}


def _render_model_stub(
    package_root: Path,
    archetype: str,
    model_name: str,
    class_suffix: str,
) -> None:
    """
    Render the Jinja stub for the PythonModel class in the target.

    Parameters
    ----------
    package_root : Path
        Root of the target package (contains model_code/archetypes).
    archetype : str
        Slug of the archetype (e.g. 'file_uri_to_tensor').
    model_name : str
        Snake_case model name.
    class_suffix : str
        Suffix for the generated class.
    """
    src = package_root / "model_code" / "archetypes"
    tpl = src / f"{archetype}_model.py.jinja"
    if not tpl.is_file():
        _LOGGER.error("Archetype stub '%s' not found at %s", archetype, tpl)
        sys.exit(2)

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(src),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    rendered = env.get_template(tpl.name).render(
        MODEL_NAME=model_name,
        MODEL_CAMEL="".join(p.capitalize() for p in model_name.split("_")),
        CLASS_SUFFIX=class_suffix,
    )
    dst = package_root / "model_code" / f"{model_name}_mlflow_model.py"
    dst.write_text(rendered, encoding="utf-8")
    _LOGGER.info("Rendered model stub → %s", dst)


def _copy_spec_stub(
    package_root: Path,
    archetype: str,
) -> None:
    """
    Copy the static model_spec stub into the package root.

    Parameters
    ----------
    package_root : Path
        Root of the target package.
    archetype : str
        Slug of the archetype (e.g. 'file_uri_to_tensor').
    """
    src = package_root / "model_code" / "archetypes" / f"{archetype}_model_spec.py"
    dst = package_root / "model_spec.py"
    if not src.is_file():
        _LOGGER.error("Spec stub not found at %s", src)
        sys.exit(2)
    shutil.copy2(src, dst)
    _LOGGER.info("Copied spec skeleton → %s", dst)


def _remove_jinja_files(package_root: Path) -> None:
    """
    Remove all Jinja templates from the target package.

    Parameters
    ----------
    package_root : Path
        Root of the target package.
    """
    for tpl in package_root.rglob("*.jinja"):
        tpl.unlink()
        _LOGGER.debug("Removed template %s", tpl)


def _remove_archetypes_dir(package_root: Path) -> None:
    """
    Remove the `model_code/archetypes` directory from the target.

    Parameters
    ----------
    package_root : Path
        Root of the target package.
    """
    archetypes_dir = package_root / "model_code" / "archetypes"
    if archetypes_dir.is_dir():
        shutil.rmtree(archetypes_dir)
        _LOGGER.info("Removed archetypes directory → %s", archetypes_dir)


def _substitute_tokens(root: Path, model_name: str) -> None:
    """
    Replace {{MODEL_NAME}} and {{MODEL_CAMEL}} in all text files under `root`.

    Parameters
    ----------
    root : Path
        Root of the target package.
    model_name : str
        Snake_case name used for tokens.
    """
    camel = "".join(part.capitalize() for part in model_name.split("_"))
    for fp in _iter_text_files(root.rglob("*")):
        txt = fp.read_text(encoding="utf-8")
        txt = txt.replace("{{MODEL_NAME}}", model_name)
        txt = txt.replace("{{MODEL_CAMEL}}", camel)
        fp.write_text(txt, encoding="utf-8")


def _parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for scaffolding.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        prog="generate_model_pkg_dir_structure.py",
        description="Scaffold an MLflow packaging directory with a Jinja archetype and spec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Snake_case name for your model package (e.g. 'cool_embedding').",
    )
    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="Existing directory under which '<model_name>_mlflow_pkg' is created.",
    )
    parser.add_argument(
        "--pkg-template",
        "--template",
        default="mlflow_template",
        help="Subfolder under pkg_templates/ to copy as the skeleton.",
    )
    parser.add_argument(
        "--model-archetype",
        default="file_uri_to_tensor",
        help="Archetype slug (matches <slug>_model.py.jinja and _model_spec.py).",
    )
    parser.add_argument(
        "--class-suffix",
        default="MLflowModel",
        help="Suffix appended to the generated class name.",
    )
    parser.add_argument(
        "--force-template-overwrite",
        action="store_true",
        help="If destination exists, back it up and overwrite.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without modifying any files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    Entry point for the scaffolding CLI.

    - Copies the package template to the target.
    - Renders the model stub and copies the spec.
    - Cleans up templates, removes `archetypes/`, and substitutes tokens.
    """
    _configure_logging()
    args = _parse_cli(argv)

    target_dir = args.target.expanduser().resolve()
    if not target_dir.is_dir():
        _LOGGER.error("Target %s is not a directory", target_dir)
        sys.exit(2)

    script_dir = Path(__file__).resolve().parent
    pkg_tpl = script_dir / "pkg_templates" / args.pkg_template
    if not pkg_tpl.is_dir():
        _LOGGER.error("Template '%s' not found", args.pkg_template)
        sys.exit(2)

    dest_pkg = target_dir / f"{args.model_name}_mlflow_pkg"
    if dest_pkg.exists():
        if args.force_template_overwrite:
            _backup_existing(dest_pkg)
        else:
            _LOGGER.error(
                "Destination exists; use --force-template-overwrite to overwrite"
            )
            sys.exit(2)

    if args.dry_run:
        _LOGGER.info("[dry-run] Would copy %s to %s", pkg_tpl, dest_pkg)
        return

    # 1) Copy entire template (excluding docs)
    shutil.copytree(
        pkg_tpl,
        dest_pkg,
        dirs_exist_ok=False,
        ignore=_ignore_docs,
    )

    # 2) Render stub & spec inside the new package
    _render_model_stub(
        package_root=dest_pkg,
        archetype=args.model_archetype,
        model_name=args.model_name,
        class_suffix=args.class_suffix,
    )
    _copy_spec_stub(
        package_root=dest_pkg,
        archetype=args.model_archetype,
    )

    # 3) Remove raw Jinja templates from the target
    _remove_jinja_files(dest_pkg)

    # 4) Remove the `archetypes/` directory now that stubs are materialized
    _remove_archetypes_dir(dest_pkg)

    # 5) Substitute tokens in all remaining text files
    _substitute_tokens(dest_pkg, args.model_name)

    _LOGGER.info("✔ Package scaffold ready at %s", dest_pkg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _LOGGER.error("Interrupted by user")
        sys.exit(1)

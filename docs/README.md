# Documentation Guide

This guide explains how to set up, build, view, and update the documentation for this project using Sphinx. It is designed for users who may not have prior experience with Sphinx.

---

## Installation

### Step 1: Create a Virtual Environment

It is recommended to use a virtual environment to isolate dependencies.

```bash
cd docs # Ensure you are in docs directory
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
```

### Step 2: Install Documentation Dependencies

Dependencies for building the documentation are listed in `pyproject.toml`. Install them using the following command:

```bash
pip install -e ".[docs]"
```

### Step 3: Install Pandoc for Notebook Support

To build documentation from Jupyter notebooks (`.ipynb` files), you need to install **Pandoc**. Pandoc is required by Sphinx extensions `nbsphinx` to convert notebooks into HTML.

#### On macOS:

```bash
brew install pandoc
```

#### On Linux (Debian/Ubuntu):

```bash
sudo apt-get update
sudo apt-get install pandoc
```

Alternatively, you can download the latest Pandoc release from the [Pandoc installation page](https://pandoc.org/installing.html).


---

## Building the Documentation

### Step 1: Generate Documentation Files

This project uses **AutoAPI** to automatically generate documentation for the Python codebase. Ensure the `autoapi_dirs` in `conf.py` points to the correct source directory (e.g., `../../src/`).

Run the following command to build the documentation:

```bash
make html
```

This will generate the HTML documentation in the `build/html` directory.

### Step 2: View the Documentation

To view the generated documentation, open the `index.html` file in the `build/html` directory in your web browser:

```bash
open build/html/index.html  # macOS/Linux
```

### Step 3: (Optional) Serve and View Documentation Locally

To view the documentation in your browser with a local HTTP server, in case a plugin using Javascript ES6 imports, follow these steps:

1. Start a local HTTP server from the `build/html` directory:
    ```bash
    python -m http.server
    ```
    Ensure that port `8000` is available on your system.

2. Open your web browser and navigate to:
    ```
    http://localhost:8000
    ```

This method allows you to browse the documentation as it would appear on a live server.
---

## Updating the Documentation

### Step 1: Editing `.rst` and `.md` Files

- **`.rst` Files**:  
  Sphinx primarily uses `.rst` (reStructuredText) files for documentation. These files allow you to structure content using headings, lists, code blocks, and more. Refer to the [Sphinx reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) for details.

- **`.md` Files**:  
  Markdown files are also supported via the `myst_parser` extension. You can write documentation in Markdown format, which is simpler and more widely used. Refer to the [MyST Markdown Guide](https://myst-parser.readthedocs.io/en/latest/) for syntax.

### Step 2: Adding New Python Modules

If you add new Python modules to the project, **AutoAPI** will automatically include them in the documentation during the next build. Ensure the `autoapi_dirs` in `conf.py` is correctly configured.

### Step 3: Rebuilding the Documentation

After making changes, rebuild the documentation using:

```bash
make clean html
```

The `clean` target ensures that old build files are removed before generating new ones.


---

## About AutoAPI

**AutoAPI** is a Sphinx extension that automatically generates documentation for Python modules. It scans the source code and creates `.rst` files for classes, functions, and modules.

### How to Use AutoAPI

1. Ensure the `autoapi_dirs` in `conf.py` points to the source directory containing your Python code.
2. Run `make html` to generate the documentation.
3. AutoAPI will create `.rst` files for all modules in the specified directory.

---

## Best Practices for Writing Documentation

- **Use Clear Headings**: Organize content with meaningful headings.
- **Add Code Examples**: Include examples to demonstrate usage.
- **Document All Functions and Classes**: Ensure every function and class has a docstring.
- **Use Cross-References**: Use Sphinx's cross-referencing features to link between sections.



## Using Mermaid Diagrams in Sphinx Documentation

Mermaid diagrams can be seamlessly integrated into your Sphinx documentation using the `sphinxcontrib-mermaid` plugin. Follow the steps below to include and render Mermaid diagrams in your project:


### Step 1: Add Mermaid Diagrams to Your Documentation

Use the `.. mermaid::` directive in your reStructuredText (`.rst`) files to include Mermaid diagrams. Below is an example:

```rst
.. mermaid::

    graph TD
        A[Start] --> B{Decision}
        B -->|Yes| C[Option 1]
        B -->|No| D[Option 2]
```

### Step 2: Build the Documentation

Build your Sphinx documentation as usual, for example:

```bash
make html
```

---

## Additional Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [MyST Markdown Guide](https://myst-parser.readthedocs.io/en/latest/)
- [Sphinx reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [AutoAPI Documentation](https://sphinx-autoapi.readthedocs.io/en/latest/)
- [Mermaid Documentation](https://mermaid-js.github.io/mermaid/)


---

## Troubleshooting

### Missing Dependencies

Ensure all dependencies are installed in your virtual environment. Reinstall them if necessary:

```bash
pip install -e ".[docs]"
```

### Build Errors

Run `make clean` before rebuilding the documentation to remove old files:

```bash
make clean
make html
```

### AutoAPI Issues

Verify that the `autoapi_dirs` in `conf.py` points to the correct source directory.
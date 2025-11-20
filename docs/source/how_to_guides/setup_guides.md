# Setup Guides

This guide will help you set up a development environment for **cz-benchmarks** on macOS, Linux, or Windows.

## Prerequisites

1. **Install Python and a Tool for Environment/Dependency Management**  
   Make sure you have [Python 3.10+](https://www.python.org/downloads/) installed.  
   There are several tools you can use for managing Python environments and dependencies:
   - [pip and venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
   - [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)
   - [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (described below)

   Choose the tool you prefer and make sure it is installed.

2. **Install Build Tools and Python Headers**  
   Some dependencies require C/C++ compilers and Python headers.
   - **macOS:** Install Xcode Command Line Tools:
     ```bash
     xcode-select --install
     ```
   - **Linux (Debian/Ubuntu):** Install compilers and Python dev headers:
     ```bash
     sudo apt-get update
     sudo apt-get install build-essential python3-dev
     ```
   - **Windows:** [Install Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (select "C++ build tools" during installation).

## Setting Up the Environment

1. **Create a Virtual Environment**  
   Using a virtual environment is strongly recommended to isolate dependencies.  
   Example using `venv`:

   ```bash
   python -m venv venv
   ```

   Activate the environment:

   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - Windows:
     ```bash
     venv\Scripts\activate
     ```

2. **Install Dependencies**  
   Install the required Python packages in editable mode with development dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

   > **Note:** On some systems (especially macOS and Linux), you may need to ensure system C/C++ build tools and Python header files are present before installing dependencies such as `hnswlib` (see above).

---

## Using `uv` for Dependency Management

[`uv`](https://docs.astral.sh/uv/) is an alternative tool that simplifies Python dependency management.

1. **Install `uv`**  
   You can use `pip` to install `uv`:

   ```bash
   pip install uv
   ```

2. **Install the Required Python Version (Optional)**  
   If your Python version doesn't match the project requirements, `uv` can help install/manage it:

   ```bash
   uv python install
   ```

3. **Sync Dependencies**  
   To install all project dependencies, including optional "extras":

   ```bash
   uv sync --all-extras
   ```

4. **Troubleshooting `hnswlib` Installations**

   If you encounter an error involving `Python.h` (e.g., `fatal error: Python.h: No such file or directory`), make sure Python development headers and static libraries are installed:
   - **Linux (Debian/Ubuntu):**
     ```bash
     sudo apt-get install python3-dev
     ```
   - **macOS:** Ensure Xcode Command Line Tools are installed (`xcode-select --install`).
   - **Windows:** Ensure Visual Studio C++ Build Tools are present.

> 💡 **Tip**: For more details, see the [official `uv` installation guide](https://docs.astral.sh/uv/getting-started/installation/).


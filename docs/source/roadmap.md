# Roadmap

👋 Welcome to the public roadmap for the cz-benchmarks package — a collaborative effort to enable the scientific community to conduct reproducible, biologically-relevant benchmarking of AI models for biology, across domains [learn more](/README.md).

The roadmap reflects our initial priorities and sequencing of work. Our goals in sharing this roadmap are to enable community engagement and contribution through transparency - without this, we know that our work here cannot be successful.

However, please note that this is a very early stage project and as such, we expect the roadmap to change depending on what is learned through code development, community discussion and engagement, and internal priorities. Changes in roadmap are subject to the current governing principles of the team (see below).


## 🙋 Priority User Needs

- Can I load data, run, and reproduce a benchmark/task?
- Can I use the benchmarks in my model developer workflow?
- How can I, as a member of the community, contribute benchmarks?
- (internal, exploratory) Can we build a benchmarking package that spans multiple modalities/domains?

---

## 🎯 Now: Core Infrastructure & Reproducibility

### In Development:
- Open-source the cz-benchmarks repository and develop PyPi package
- Enable running cz-benchmarks on an initial set of models, datasets, and tasks
- Publish developer-facing documentation to run benchmarks on other models (this first version will take some effort from the developer to get set up; we aim to make this more seamless in the future, see below)
- 🔬 Initial domain focus: single-cell transcriptomics

### 📋 Candidate release tags:
- v0.1: Open alpha (unstable) repo for targeted developer preview
- v0.2: Repo generates benchmark run results that can be ingested for visualization on CZI’s Virtual Cell Platform UI

---

## 🔍 Next: Support Developer Workflow & Early Contribution Workflow

### Possible Candidates:
- Refactor to better support model developer workflows, enabling more seamless use of custom assets (models, datasets)
- Refactor to adopt standardized model packaging (MLflow)
- Refactor CLI to create a more user-friendly interface for running benchmarks
- Enable initial contributors to add (alpha format):
  - New models
  - New datasets, tasks, and metrics
- Expand suite of community-driven assets to incorporate [working group](https://virtualcellmodels.cziscience.com/micro-pub/jamboree-launches-working-group) recommendations
- Develop and publish tutorial notebooks and developer workflow examples
- Visualize benchmarking results as part of CZI’s [Virtual Cell Platform](http://virtualcellmodels.cziscience.com)
- 🔬 Expand domain focus to: Imaging models (e.g. cell morphology)

---

## 🚀 Future Ideas:
- Refine contribution workflow for comprehensive suite of [assets – models, datasets, tasks, metrics]((docs/source/assets.md))
- NVIDIA tutorials to assist developers in leveraging cz-benchmarks in their workflows
- Benchmarking on held-out datasets via hosted inference
- 🔬 Expand domain focus: DNA-based models, spatial transcriptomics

---

## Roadmap Governance

Our goal is to work towards a community developed benchmarking resource that will be useful for the scientific community. In the short term, to get an alpha release initiated and stable, we currently operate using a simple governance model as follows:

- **Roadmap Leads**: Katrina, Olivia (CZI) and Laksshman, TJ (NVIDIA)
- **Tech Leads**: Sanchit, Andrew (CZI) and Ankit, Michelle (NVIDIA)

Roadmap alignment and decision making are completed by the Roadmap Leads, in close collaboration with the Tech Leads, by consensus wherever possible. CZI SciTech currently maintains ownership of the repository and holds final decision-making authority. We will be working in close collaboration with NVIDIA to execute the roadmap and will continue to evolve the governance structure based on project needs, community growth, and resourcing. Guidelines and governance for included assets are available [here](docs/source/assets.md).

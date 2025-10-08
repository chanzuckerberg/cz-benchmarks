# Roadmap

👋 Welcome to the public roadmap for the cz-benchmarks package — a collaborative effort to enable the scientific community to conduct reproducible, biologically-relevant benchmarking of AI models for biology, across domains.

The roadmap reflects our initial priorities and sequencing of work. Our goal in sharing this roadmap is to enable community engagement and contribution through transparency; without this, we know that our work here cannot be successful.

Please note that this is an early stage project and as such, we expect the roadmap to change depending on what is learned through code development, community engagement, and internal priorities. Changes in roadmap are subject to the current governing principles of the team (see below).


## 🙋 Priority User Needs

- Can I load data, run, and reproduce a benchmark?
- Can I use the benchmarks in my model developer workflow?
- How can I, as a member of the community, contribute benchmarks?
- (internal, exploratory) Can we build a benchmarking package that spans multiple modalities/domains required for Virtual Cell modeling?


## 🎯 Launched & Live: Core Infrastructure & Reproducibility

The [cz-benchmarks](https://github.com/chanzuckerberg/cz-benchmarks) repository is open sourced and available as a [PyPi package](https://pypi.org/project/cz-benchmarks/). 🎉 With this first major release, we’ve…

- Focused cz-benchmarks’ scope on benchmarking tasks and metrics. As the first step in this work, we pulled out individual models from being integrated directly within cz-benchmarks. This separation of models from datasets, tasks & metrics enables flexible usage of the cz-benchmarks tasks and datasets across various model configurations, ensures that tasks and metrics stay open-sourced and community-driven, and follows industry practice seen in other benchmark tools.
  - CZI’s [Virtual Cells Platform](http://virtualcellmodels.cziscience.com) is already a natural home for curated models & datasets. As such, we’ve added complementary support in the [VCP CLI](https://pypi.org/project/vcp-cli/) to support running, viewing, and reproducing our [public benchmarking results](https://virtualcellmodels.cziscience.com/benchmarks) - powered by cz-benchmarks and the core resources on the Virtual Cells Platform.
  - By decoupling VCP from cz-benchmarks, model developers can utilize these tasks and benchmarks to evaluate their own models during development.
- Published six foundational tasks that reflect common needs in single-cell biology:
  - Cell clustering
  - Cell type classification
  - Cross-species integration
  - Cross-species disease label transfer (contributed by a community working group)
  - Perturbation modeling (contributed by NVIDIA)
  - Sequential ordering (contributed by the Allen Institute)
- Published six cz-benchmarks compatible benchmark datasets that support evaluations across the above listed tasks
- Published tutorial notebooks and developer workflow examples
- 🔬 Initial domain focus: single-cell transcriptomics

### 📋 Candidate release tags

- v0.10: Improved API for use in model development workflows. Removed models from being directly embedded in cz-benchmarks, but kept compatible.


## 🔍 Next: Support Developer Workflow & Early Contribution Workflow

### In Development

- Incorporate feedback on cz-benchmarks from community use
- Following up on the work to pull models out of cz-benchmarks and have them be compatible with the package, we’ll be following the same pattern with benchmark datasets. At the end of this work, the goal is to have cz-benchmarks stay focused on metrics & tasks, while being complementary with the VCP curated and hosted models & datasets.
- Expand suite of community-driven assets to incorporate additional [working group](https://virtualcellmodels.cziscience.com/micro-pub/jamboree-launches-working-group) recommendations and NVIDIA
- Expand suite of transcriptomic perturbation benchmarks
- 🔬 Explore expanding domain focus to: Imaging (specifically, image representation learning, virtual staining, and dynamic imaging) and DNA models. If you’re interested in contributing to early stage work in these areas, please reach out to us at [virtualcellmodels@chanzuckerberg.com](mailto:virtualcellmodels@chanzuckerberg.com).

---

## 🚀 Future Ideas

- Refine contribution workflow for [tasks and metrics](https://chanzuckerberg.github.io/cz-benchmarks/assets.html) into cz-benchmarks
- NVIDIA tutorials to assist developers in leveraging cz-benchmarks in their workflows
- Benchmarking on held-out datasets via hosted inference

---

## Roadmap Governance

Our goal is to work towards a community-developed benchmarking resource that will be useful for the scientific community. In the short term, to get an alpha release initiated and stable, we currently operate using a simple governance model as follows:

- Roadmap Leads: Katrina Kalantar, Olivia Holmes (CZI) and Laksshman Sundaram, TJ Chen (NVIDIA)
- Tech Leads: Sanchit Gupta, Andrew Tolopko (CZI) and Ankit Sethia, Michelle Gill (NVIDIA)

Roadmap alignment and decision making are completed by the Roadmap Leads, in close collaboration with the Tech Leads, by consensus wherever possible. CZI SciTech currently maintains ownership of the repository and holds final decision-making authority. We will be working in close collaboration with NVIDIA to execute the roadmap and will continue to evolve the governance structure based on project needs, community growth, and resourcing. Guidelines and governance for included assets are available [here](https://chanzuckerberg.github.io/cz-benchmarks/assets.html).

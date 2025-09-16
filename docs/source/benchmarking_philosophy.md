# Benchmarking Philosophy

## Guiding Principles for Model Evaluation

We believe that benchmarking pushes the field of model development toward biological relevance and utility. To achieve this, we’ve established the following principles to guide our design and implementation of benchmarks for AI models in biology.

### 1. Evaluation Should Challenge Models

- We intentionally design or select evaluation datasets that stress-test model generalization and robustness and avoid model overfitting.
- Models may be evaluated on data that differs in modality, species, or context from what they were trained on.
- To prevent performance inflation and metric saturation, we aim to regularly update benchmark datasets and tasks.

### 2. Deep-Learning Models Will Be Evaluated Across Diverse Tasks

- Because large deep-learning models aim to be broadly useful, we evaluate them on a wide range of biologically relevant tasks, regardless of whether they were explicitly trained for those tasks. This ensures we capture their true generalization ability and relevance to the biology community.

### 3. Fine-Tuning Is Community-Led, Not Centralized

- The CZI benchmarking team does not fine-tune models internally for every task. However, we welcome community-submitted fine-tuned models, especially when paired with insights as to why fine-tuning improves performance for specific biological tasks. These contributions can be integrated into the [Virtual Cells Platform](https://virtualcellmodels.cziscience.com/) for standardized evaluation.

### 4. Prioritizing Biological Impact

- We aim to support benchmarks that matter most to biologists. This means prioritizing biological relevance, translational value, and scientific utility.

### 5. Tasks and Evaluation Datasets Will be Made Available to The Community Early

- We recognize that benchmark tasks and evaluation datasets are valuable community resources. To support open science and accelerate model development, we will release tasks and evaluation datasets as soon as they are ready, even if [benchmarking results](https://virtualcellmodels.cziscience.com/benchmarks) are not yet available.

### 6. Community Contributions are Prioritized

The best way to define valuable benchmarks is through community participation. To that end:

- We create and support working groups of domain experts in biology and machine learning
- We seek partnership! Work with us to contribute benchmarking assets. Right now we are prioritizing assets in the single-cell transcriptomic and perturbation modeling domains, and aim to pilot expanding to DNA and dynamic imaging by 2026. If you are working in any of these domains and are interested in partnering with us, email us at [virtualcellmodels@chanzuckerberg.com](mailto:virtualcellmodels@chanzuckerberg.com). 

### 7. Feedback Loops Are Built-In

We encourage feedback from the community at every stage:  
– Task definitions  
– Metric choices  
– Dataset selection  
– Interpretation of benchmarking results

Please reach out via this [feedback form](https://airtable.com/appd6ZLxfAOLcfNcs/paggB4T2aE2J5kIJs/form?hide_user_id=true&hide_device_id=true&hide_amplitude_id=true&prefill_benchmark_id=cell-clustering&hide_benchmark_id=true) or email us at [virtualcellmodels@chanzuckerberg.com](mailto:virtualcellmodels@chanzuckerberg.com).  to help improve the benchmarking platform.

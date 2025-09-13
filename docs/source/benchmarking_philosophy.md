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
- We seek partnership! Work with us to contribute benchmarking assets. Right now we are prioritizing assets in the single-cell transcriptomic and perturbation modeling domains, and aim to pilot expanding to DNA and dynamic imaging by 2026. If you are working in any of these domains and are interested in partnering with us, email us at [email]  

### 7. Feedback Loops Are Built-In

We encourage feedback from the community at every stage:  
– Task definitions  
– Metric choices  
– Dataset selection  
– Interpretation of benchmarking results

Please reach out via this [feedback form](https://airtable.com/appd6ZLxfAOLcfNcs/paggB4T2aE2J5kIJs/form?hide_user_id=true&hide_device_id=true&hide_amplitude_id=true&prefill_benchmark_id=cell-clustering&hide_benchmark_id=true) or email us at [email] to help improve the benchmarking platform.

---

### FOR FUTURE RELEASES: 

#### Technical Standards

- Something about data leakage/OOD/evaluation dataset curation
- We categorize and support benchmark runs of the following:  
    - **GREEN [ideal state]:** Eval dataset <> model variant where data has NOT been seen in pretraining
    - **YELLOW [current industry practice]:** Eval dataset <> model variant where data has been seen in pretraining, but not in task training
    - **RED:** eval dataset <> model variant where data has been seen in task training [we do not support these pairings]
    - **GRAY:** Eval dataset <> model variant where it is unclear due to insufficient model documentation whether data has been seen in pretraining 
- For sample-level predictions, where possible, we ensure that samples from the same donor are not represented in both training and evaluation 
- [TF-specific] Where possible, gene orthologs are used to evaluate models that do not innately support an out of distribution species. 

##### What methods/models we include

- We include foundational models
- Do we want to limit by “feasability”, i.e. if a model takes more than XYZ hours to run on dataset A, we do not include
- [EC] scratch space to double check this
    - scVI: NO
    - Geneformer: yes
    - UCE: yes
    - scGPT: yes
    - TF[sapeins,metazoa,exemplar]: yes
    - AIDO: yes

- Do we compare against current SoA approaches for doing the task, e.g. if bulk deconv, SoA approach is something like CIBERSortx? If cell type annotation, SoA approach is something like reference-based annotation?

##### Baseline approaches

- Why we choose specific methods as the baseline
- Ideally, we would include a random baseline → I’m not sure this will be ready by first launch so maybe we don’t include then?
- We support two types of baselines:
    - **Random baseline:** this represents the performance of the task at random [not yet supported across the board]
    - **Simple baseline:** this represents the simplest possible industry-accepted workflow [this needs work to explain]

##### Data size guidelines 

- Will this need to be domain specific? (Imaging vs genomics)

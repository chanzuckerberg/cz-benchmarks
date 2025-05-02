
# Assets

While cz-benchmarks will eventually be cross-compatible with assets hosted via the [Virtual Cells Platform](#) as well as user-defined and contributed assets [roadmap](../roadmap.md), we provide interim documentation below. 

Note: Supported assets can be listed via the [CLI](../quick_start.md)

## Models

| Model Name     | Description                                                                             | VCP Model Card |
| -------------- | --------------------------------------------------------------------------------------- | -------------- |
| **SCVI**       | Probabilistic model using variational autoencoders for transcriptomic data              | [Link](#)           |
| **SCGPT**      | Transformer model trained for general-purpose transcriptomic embedding                  | [Link](#)           |
| **Geneformer** | Transformer pre-trained on bulk and single-cell transcriptomes for functional inference | [Link](#)           |
| **scGenePT**   | Gene perturbation transformer trained to predict gene knockout response                 | [Link](#)           |
| **UCE**        | Universal Cell Embedding model trained across species with protein-guided embeddings    | [Link](#)           |


## Datasets

| Dataset Name                     | Description                                          | Organism         | VCP Data Card |
| -------------------------------- | ---------------------------------------------------- | ---------------- | ------------- |
| **adamson_perturb**              | Gene perturbation CRISPR dataset from Adamson et al. | Human            | [Link](#)          |
| **norman_perturb**               | Gene perturbation CRISPR dataset from Norman et al.  | Human            | [Link](#)          |
| **dixit_perturb**                | Gene perturbation CRISPR dataset from Dixit et al.   | Human            | [Link](#)          |
| **replogle_k562_perturb**        | Gene perturbation CRISPR dataset in K562 cells       | Human            | [Link](#)          |
| **replogle_rpe1_perturb**        | Gene perturbation CRISPR dataset in RPE1 cells       | Human            | [Link](#)          |
| **human_spermatogenesis**        | Single-cell dataset of human spermatogenesis         | Human            | [Link](#)          |
| **mouse_spermatogenesis**        | Single-cell dataset of mouse spermatogenesis         | Mouse            | [Link](#)          |
| **rhesus_macaque_spermatogenesis** | Single-cell dataset of rhesus macaque spermatogenesis | Rhesus Macaque   | [Link](#)          |
| **gorilla_spermatogenesis**      | Single-cell dataset of gorilla spermatogenesis       | Gorilla          | [Link](#)          |
| **chimpanzee_spermatogenesis**   | Single-cell dataset of chimpanzee spermatogenesis    | Chimpanzee       | [Link](#)          |
| **marmoset_spermatogenesis**     | Single-cell dataset of marmoset spermatogenesis      | Marmoset         | [Link](#)          |
| **chicken_spermatogenesis**      | Single-cell dataset of chicken spermatogenesis       | Chicken          | [Link](#)          |
| **opossum_spermatogenesis**      | Single-cell dataset of opossum spermatogenesis       | Opossum          | [Link](#)          |
| **platypus_spermatogenesis**     | Single-cell dataset of platypus spermatogenesis      | Platypus         | [Link](#)          |
| **tsv2_bladder**                 | Tabula Sapiens v2 bladder tissue                     | Human            | [Link](#)          |
| **tsv2_blood**                   | Tabula Sapiens v2 blood tissue                       | Human            | [Link](#)          |
| **tsv2_bone_marrow**             | Tabula Sapiens v2 bone marrow tissue                 | Human            | [Link](#)          |
| **tsv2_ear**                     | Tabula Sapiens v2 ear tissue                         | Human            | [Link](#)          |
| **tsv2_eye**                     | Tabula Sapiens v2 eye tissue                         | Human            | [Link](#)          |
| **tsv2_fat**                     | Tabula Sapiens v2 fat tissue                         | Human            | [Link](#)          |
| **tsv2_heart**                   | Tabula Sapiens v2 heart tissue                       | Human            | [Link](#)          |
| **tsv2_large_intestine**         | Tabula Sapiens v2 large intestine tissue             | Human            | [Link](#)          |
| **tsv2_liver**                   | Tabula Sapiens v2 liver tissue                       | Human            | [Link](#)          |
| **tsv2_lung**                    | Tabula Sapiens v2 lung tissue                        | Human            | [Link](#)          |
| **tsv2_lymph_node**              | Tabula Sapiens v2 lymph node tissue                  | Human            | [Link](#)          |
| **tsv2_mammary**                 | Tabula Sapiens v2 mammary tissue                     | Human            | [Link](#)          |
| **tsv2_muscle**                  | Tabula Sapiens v2 muscle tissue                      | Human            | [Link](#)          |
| **tsv2_ovary**                   | Tabula Sapiens v2 ovary tissue                       | Human            | [Link](#)          |
| **tsv2_prostate**                | Tabula Sapiens v2 prostate tissue                    | Human            | [Link](#)          |
| **tsv2_salivary_gland**          | Tabula Sapiens v2 salivary gland tissue              | Human            | [Link](#)          |
| **tsv2_skin**                    | Tabula Sapiens v2 skin tissue                        | Human            | [Link](#)          |
| **tsv2_small_intestine**         | Tabula Sapiens v2 small intestine tissue             | Human            | [Link](#)          |
| **tsv2_spleen**                  | Tabula Sapiens v2 spleen tissue                      | Human            | [Link](#)          |
| **tsv2_stomach**                 | Tabula Sapiens v2 stomach tissue                     | Human            | [Link](#)          |
| **tsv2_testis**                  | Tabula Sapiens v2 testis tissue                      | Human            | [Link](#)          |
| **tsv2_thymus**                  | Tabula Sapiens v2 thymus tissue                      | Human            | [Link](#)          |
| **tsv2_tongue**                  | Tabula Sapiens v2 tongue tissue                      | Human            | [Link](#)          |
| **tsv2_trachea**                 | Tabula Sapiens v2 trachea tissue                     | Human            | [Link](#)          |
| **tsv2_uterus**                  | Tabula Sapiens v2 uterus tissue                      | Human            | [Link](#)          |
| **tsv2_vasculature**             | Tabula Sapiens v2 vasculature tissue                 | Human            | [Link](#)          |


## Tasks & Metrics

| Task Name                     | Description                                                                         | Associated Metrics                                             |
| ----------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| **Clustering**                | Cluster cells in embedding space and evaluate against known labels (e.g. cell type) | Adjusted Rand Index (ARI), Normalized Mutual Information (NMI) |
| **Embedding**                 | Measure separation of cell types in embedding space                                 | Silhouette Score                                               |
| **Label Prediction**          | Use classifiers (e.g. logistic regression, KNN) to predict metadata from embeddings | Accuracy, F1, Precision, Recall, AUROC                         |
| **Integration**               | Assess batch mixing and label retention across batches                              | Entropy per Cell, Batch Silhouette Score                       |
| **Perturbation**              | Compare predicted vs ground-truth expression shifts under perturbation              | MSE, Pearson R², Jaccard Index                                 |
| **Cross-Species Integration** | Evaluate whether embeddings can align multiple species in a shared space            | Entropy per Cell, Silhouette Score                             |


## Compatibility Matrix

| Model      | Clustering | Embedding | Label Prediction | Integration | Perturbation | Cross-Species |
| ---------- | ---------- | --------- | ---------------- | ----------- | ------------ | ------------- |
| SCVI       | ✓          | ✓         | ✓                | ✓           |              |               |
| SCGPT      | ✓          | ✓         | ✓                |             |              |               |
| Geneformer | ✓          | ✓         | ✓                |             |              |               |
| scGenePT   |            |           |                  |             | ✓            |               |
| UCE        | ✓          | ✓         | ✓                |             |              | ✓             |





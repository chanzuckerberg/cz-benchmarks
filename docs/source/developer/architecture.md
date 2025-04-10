## Architecture

The core components:

-   **Datasets**: Input format validation, AnnData-backed
-   **Models**: Encapsulated in Docker containers
-   **Tasks**: Define evaluations (e.g., clustering, perturbation)
-   **Metrics**: Registered using `MetricRegistry`
-   **Runner**: Handles containerized execution with automatic serialization


```
[User Input]
    ↓
[Hydra Config] --> [Dataset Loader] --> [ContainerRunner] --> [Model Docker Image]
                                                           ↓
                                                [Outputs] --> [Task] --> [MetricRegistry]
```

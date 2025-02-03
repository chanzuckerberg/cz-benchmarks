import bentoml
from bentoml.io import JSON

from czibench.datasets.utils import load_dataset

from .model import SCVI


class SCVIRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        self.model = SCVI()

    @bentoml.Runnable.method(batchable=False)
    def run(self, config_path: str) -> dict:
        # Load dataset
        dataset = load_dataset("example", config_path=config_path)
        
        # Run inference
        result = self.model.run(dataset)
        return result

svc = bentoml.Service(
    "scvi_service",
    runners=[bentoml.Runner(SCVIRunnable)],
)

@svc.api(input=JSON(pydantic_model=None), output=JSON(pydantic_model=None))
async def run_inference(input_data: dict) -> dict:
    """
    Run inference using the SCVI model.
    Input data should contain config_path.
    """
    config_path = input_data.get("config_path", "custom.yaml")
    runner = svc.runners["scvirunnable"]
    return await runner.run.async_run(config_path)     config_path = input_data.get("config_path", "custom.yaml")
    runner = svc.runners["scvirunnable"]
    return await runner.run.async_run(config_path) 
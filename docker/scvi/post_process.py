from pathlib import Path
import json
import czbenchmarks # might be needed

# TODO: What should the output be? A Python type or just a file?
# TODO: Should cz-benchmarks write the output to a file?
def post_process(prediction: Any, output_dir: Path) -> Path: # czbenchmarks.types.Embedding
    output_path = output_dir / Path("prediction_output.json")
    with open(output_path, "w") as f:
        json.dump(prediction, f)
    return output_path


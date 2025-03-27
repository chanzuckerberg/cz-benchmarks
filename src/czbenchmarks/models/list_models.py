from .types import ModelType


def get_model_names() -> list[str]:
    """Returns a list of all model names from ModelType enum."""
    return [model.name for model in ModelType]


if __name__ == "__main__":
    # Example usage
    model_names = get_model_names()
    print("Available models:")
    for name in model_names:
        print(f"- {name}") 
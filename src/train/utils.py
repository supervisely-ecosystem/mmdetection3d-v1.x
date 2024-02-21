import yaml
import os


def load_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def extract_model_name(cfg_model):
    return cfg_model.split("/")[-1].replace(".py", "")


def find_weights_url(model_index, cfg_model):
    model_name = extract_model_name(cfg_model)
    model_index_data = load_yaml(model_index)

    for relative_path in model_index_data["Import"]:
        metafile_path = os.path.join(os.path.dirname(model_index), relative_path)
        metafile_data = load_yaml(metafile_path)

        for model in metafile_data.get("Models", []):
            if model["Name"] == model_name:
                return str(model.get("Weights"))

    raise ValueError(f"Weights for model {model_name} not found in {model_index}")

import importlib
import os

MODEL_REGISTRY = {}

def build_model(args):
    model = None
    model_type = getattr(args, "model", None)

    if model_type in MODEL_REGISTRY:
        model = MODEL_REGISTRY[model_type]

    assert model is not None, (
        f"Could not infer model type from {model_type}. "
        f"Available models: "
        + str(MODEL_REGISTRY.keys())
        + " Requested model type: "
        + model_type
    )

    return model.build_model(args)

def register_model(name):
    
    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            print(name)
            raise ValueError(f"Cannot register duplicate model ({name})")

        MODEL_REGISTRY[name] = cls

        return cls

    return register_model_cls

def import_models(models_dir, namespace):
    for file in os.listdir(models_dir):
        path = os.path.join(models_dir, file)
        if 'modules' not in path:
            if (
                not file.startswith("_")
                and not file.startswith(".")
                and (file.endswith(".py") or os.path.isdir(path))
            ):
                model_name = file[: file.find(".py")] if file.endswith(".py") else file
                importlib.import_module(namespace + "." + model_name)
        else:
            for file_ in os.listdir(path):
                if not file.startswith("_"):
                    model_name = file_[: file_.find(".py")] if file_.endswith(".py") else file_
                    importlib.import_module(namespace + "." + 'modules' + "." + model_name)

# automatically import any Python files in the models/directory
models_dir = os.path.dirname(__file__)
import_models(models_dir, "models")
"""Project settings: dynamic catalog registration."""
from pathlib import Path

import yaml
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog, MemoryDataset
from kedro_datasets.pandas import CSVDataset
from kedro_datasets.pickle import PickleDataset
from kedro.config import OmegaConfigLoader  # noqa: E402


class RegisterDynamicDatasets:
    """Hook: add model-specific outputs into DataCatalog."""

    @hook_impl
    def register_catalog(self, catalog: DataCatalog) -> DataCatalog:
        # 1. read parameter file
        conf_base = Path(__file__).resolve().parents[2] / "conf" / "base"
        with open(conf_base / "parameters_model_training.yml", encoding="utf-8") as fp:
            params: dict[str, object] = yaml.safe_load(fp)

        for model_key in params["models"]:
            suffix = model_key.lower()

            # best model pickle
            catalog.add(
                f"best_model_{suffix}",
                PickleDataset(filepath=f"data/06_models/{suffix}_model.pkl"),
            )

            # validation accuracy kept in memory
            catalog.add(f"val_accuracy_{suffix}", MemoryDataset())

            # optional test predictions
            catalog.add(
                f"y_pred_{suffix}",
                CSVDataset(filepath=f"data/07_model_output/{suffix}_pred.csv"),
            )

        return catalog

# Kedro discovers hooks via this iterable
HOOKS = (RegisterDynamicDatasets(),)

CONFIG_LOADER_CLASS = OmegaConfigLoader
# Keyword arguments to pass to the `CONFIG_LOADER_CLASS` constructor.
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    "config_patterns": {
        "spark": ["spark*", "spark*/**"],
    }
}

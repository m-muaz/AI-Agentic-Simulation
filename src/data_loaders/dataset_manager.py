# src/data_loaders/dataset_manager.py
from pathlib import Path
import pandas as pd
from .balboni_data import build_balboni_panel


class DatasetManager:
    def __init__(self, raw_zip_path: str | Path = "data/balboni/dataverse_files.zip"):
        self.raw_zip = Path(raw_zip_path)
        self.processed_dir = Path("data/balboni/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.processed_file = self.processed_dir / "balboni_panel.csv"
        self.panel = None
        self._load_or_build()

    def _load_or_build(self):
        if self.processed_file.exists():
            self.panel = pd.read_csv(self.processed_file)
        else:
            self.panel = build_balboni_panel(self.raw_zip)
            self.panel.to_csv(self.processed_file, index=False)

    def get_panel(self):
        return self.panel.copy()


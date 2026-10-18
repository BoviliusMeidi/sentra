import os
import glob
import logging
from datetime import datetime
from pathlib import Path
from src.config import SAVED_MODELS_DIR

class ModelRegistry:
    @staticmethod
    def generate_model_path(subdistrict: str, band: str, custom_name: str = None) -> str:
        base_dir = Path(SAVED_MODELS_DIR) / subdistrict / band
        base_dir.mkdir(parents=True, exist_ok=True)

        if custom_name and custom_name.strip():
            safe_name = "".join(c if c.isalnum() else "_" for c in custom_name)
            filename = f"{safe_name}.pkl"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{band}_{subdistrict}_{timestamp}.pkl"

        return str(base_dir / filename)

    @staticmethod
    def get_available_models(subdistrict: str, band: str) -> list:
        search_path = Path(SAVED_MODELS_DIR) / subdistrict / band / "*.pkl"
        model_files = glob.glob(str(search_path))

        model_files.sort(key=os.path.getmtime, reverse=True)

        history = []
        for file in model_files:
            file_name = os.path.basename(file)
            mod_time = datetime.fromtimestamp(os.path.getmtime(file)).strftime('%Y-%m-%d %H:%M:%S')
            history.append({
                "model_name": file_name,
                "file_path": file,
                "created_at": mod_time
            })

        logging.info(f"Found {len(history)} models for {subdistrict} - {band}")
        return history
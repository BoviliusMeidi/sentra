import shutil
from pathlib import Path

class TempWorkspaceManager:
    """Manages temporary files uploaded via Streamlit."""

    def __init__(self, base_temp_dir: str = "data/custom_workspace"):
        self.base_temp_dir = Path(base_temp_dir)
        self.base_temp_dir.mkdir(parents=True, exist_ok=True)

    def save_uploaded_file(self, uploaded_file, session_id: str, custom_name: str = None) -> str:
        if uploaded_file is None:
            return None

        session_dir = self.base_temp_dir / session_id
        session_dir.mkdir(exist_ok=True)

        file_name = custom_name if custom_name else uploaded_file.name
        file_path = session_dir / file_name

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return str(file_path)

    def cleanup_session(self, session_id: str):
        session_dir = self.base_temp_dir / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)
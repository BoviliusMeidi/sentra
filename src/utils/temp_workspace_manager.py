import shutil
from pathlib import Path

class TempWorkspaceManager:
    """Manages temporary files uploaded via Streamlit."""

    def __init__(self, base_temp_dir: str = "data/temp_uploads"):
        self.base_temp_dir = Path(base_temp_dir)
        self.base_temp_dir.mkdir(parents=True, exist_ok=True)

    def save_uploaded_file(self, uploaded_file, session_id: str) -> str:
        if uploaded_file is None:
            return None

        session_dir = self.base_temp_dir / session_id
        session_dir.mkdir(exist_ok=True)

        file_path = session_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return str(file_path)

    def cleanup_session(self, session_id: str):
        """Deletes the temporary folder after training is complete."""
        session_dir = self.base_temp_dir / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)
"""Copy shared/workshop_common.py into every framework folder.

Each folder keeps its own copy so it works on its own (copy one folder to Colab and it runs).
Edit shared/workshop_common.py, then run:  python tools/sync_common.py
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "shared" / "workshop_common.py"
FOLDERS = ["01_langgraph", "02_crewai", "03_openai_agents_sdk", "04_google_adk",
           "05_microsoft_agent_framework", "06_mlflow", "07_a2a_multi_agent", "08_monitoring", "09_mcp"]

for folder in FOLDERS:
    shutil.copy(SOURCE, ROOT / folder / "workshop_common.py")
    print(f"copied -> {folder}/workshop_common.py")

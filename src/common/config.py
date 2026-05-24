import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if it exists
load_dotenv()

# Root directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
RAW_DIR = Path(os.getenv("RAW_DIR", DATA_DIR / "raw"))
INTERIM_DIR = Path(os.getenv("INTERIM_DIR", DATA_DIR / "interim"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", DATA_DIR / "processed"))

# API Keys
SMHI_API_KEY = os.getenv("SMHI_API_KEY")
FMI_API_KEY = os.getenv("FMI_API_KEY")
SYKE_API_KEY = os.getenv("SYKE_API_KEY")

# Create directories if they don't exist
for path in [DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR]:
    path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(f"Base Directory: {BASE_DIR}")
    print(f"Data Directory: {DATA_DIR}")

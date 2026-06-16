from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

MAX_FILE_SIZE = 50 * 1024 * 1024   # 50 MB
MAX_DURATION_SECONDS = 600         # 10 minutes
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
RESULT_TTL_SECONDS = 3600           # 1 hour
ANALYSIS_TIMEOUT_SECONDS = 300     # 5 minutes

SUPPORTED_LANGUAGES = {"zh", "en", "auto"}
TARGET_SR = 24000

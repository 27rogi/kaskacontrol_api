import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIRECTORY = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIRECTORY / ".env")


def configured_path(name: str, default: str) -> Path:
    path = Path(os.getenv(name, default))
    return path if path.is_absolute() else ROOT_DIRECTORY / path


HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
RELOAD = os.getenv("RELOAD", "false").lower() == "true"
CORS_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "").split(",") if origin.strip()]
CORS_ORIGIN_REGEX = os.getenv("CORS_ORIGIN_REGEX", r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$").strip()
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")
DATABASE_FILE = configured_path("DATABASE_FILE", "sqlite.db")
UPLOAD_DIRECTORY = configured_path("UPLOAD_DIRECTORY", "uploads")
FONT_PATH = configured_path("FONT_PATH", "src/MartianMonoNerdFontMono-Bold.ttf")
VIOLATION_IMAGE_URL = "/uploads/violations"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "524288000"))
VIDEO_FRAME_STEP = max(1, int(os.getenv("VIDEO_FRAME_STEP", "5")))
VIDEO_WORKERS = max(1, int(os.getenv("VIDEO_WORKERS", "2")))
VIDEO_ENCODER = os.getenv("VIDEO_ENCODER", "h264_nvenc")
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "").strip()
YOLO_WEIGHTS = configured_path("YOLO_WEIGHTS", "") if YOLO_WEIGHTS else None

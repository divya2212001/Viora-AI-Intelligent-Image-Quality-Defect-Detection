import os
from pathlib import Path

from dotenv import load_dotenv


# Find the project root:
# backend/
# ├── app/
# └── .env
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")
load_dotenv()


class Settings:
    APP_NAME: str = os.getenv(
        "APP_NAME",
        "Viora AI - Image Quality & Defect Detection",
    )

    APP_VERSION: str = os.getenv(
        "APP_VERSION",
        "1.0.0",
    )

    MONGODB_URI: str = os.getenv(
        "MONGODB_URI",
        "mongodb://localhost:27017",
    )

    DATABASE_NAME: str = os.getenv(
        "DATABASE_NAME",
        "image_quality_db",
    )

    MAX_FILE_SIZE_MB: int = int(
        os.getenv("MAX_FILE_SIZE_MB", "10")
    )

    MAX_FILE_SIZE_BYTES: int = (
        MAX_FILE_SIZE_MB * 1024 * 1024
    )

    UPLOAD_DIR: Path = Path(
        os.getenv(
            "UPLOAD_DIR",
            str(BASE_DIR / "uploads"),
        )
    )

    ALLOWED_IMAGE_TYPES: tuple[str, ...] = (
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
    )

    MODEL_NAME: str = os.getenv(
        "MODEL_NAME",
        "ImageQualityNet",
    )

    MODEL_VERSION: str = os.getenv(
        "MODEL_VERSION",
        "0.1.0",
    )

    # Comma-separated browser origins permitted to call this API.
    CORS_ORIGINS: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            ",".join(
                (
                    "http://localhost:5173",
                    "http://127.0.0.1:5173",
                    "http://viora-ai-intelligent-image-quality.vercel.app",
                    "https://viora-ai-intelligent-image-quality.vercel.app",
                    "https://viora-ai-intelligent-image-quality.onrender.com",
                    
                )
            ),
        ).split(",")
        if origin.strip()
    )


settings = Settings()

settings.UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

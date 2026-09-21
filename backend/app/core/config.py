"""Application configuration via environment variables."""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent
    APP_NAME: str = "FlyMind API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = ["*"]

    DATA_ROOT: Path | None = Field(default=None)
    MODELS_DIR: Path | None = Field(default=None)
    REPORTS_DIR: Path | None = Field(default=None)

    model_config = {"env_prefix": "FLYMIND_", "env_file": ".env"}

    @property
    def SRC_DIR(self) -> Path:
        return self.PROJECT_ROOT / "src"

    @property
    def PROCESSED_DIR(self) -> Path:
        return self.PROJECT_ROOT / "data" / "processed"

    @property
    def LP_DIR(self) -> Path:
        return self.PROJECT_ROOT / "data" / "processed" / "link_prediction"

    def model_post_init(self, __context) -> None:
        if self.DATA_ROOT is None or not self.DATA_ROOT.exists():
            object.__setattr__(self, "DATA_ROOT", self.PROJECT_ROOT / "data")
        if self.MODELS_DIR is None or not self.MODELS_DIR.exists():
            object.__setattr__(self, "MODELS_DIR", self.PROJECT_ROOT / "models")
        if self.REPORTS_DIR is None or not self.REPORTS_DIR.exists():
            object.__setattr__(self, "REPORTS_DIR", self.PROJECT_ROOT / "results" / "reports")


settings = Settings()

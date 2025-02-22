import enum
import logging
import sys
from pathlib import Path
from typing import Any
from pydantic import BaseSettings, root_validator
from pydantic.networks import PostgresDsn


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class COL_NAMES(str):
    user_id = "user_id"
    user_name = "user_name"
    item_id = "item_id"
    anime_id = "anime_id"
    score = "score"
    favorite = "favorite"
    status = "status"
    progress = "progress"


class PathsSettings(BaseSettings):
    root: Path = PROJECT_ROOT
    data: Path = DATA_DIR
    interactions_raw: Path = DATA_DIR / "interactions_raw"
    interactions_score_favorite: Path = DATA_DIR / "interactions_score_favorite.parquet"
    user_name_id: Path = DATA_DIR / "user_name_id.parquet"


class PostgresSettings(BaseSettings):
    host: str
    port: int
    db: str
    user: str
    password: str
    dsn: PostgresDsn = None
    dsn_async: PostgresDsn = None

    @root_validator(skip_on_failure=True, allow_reuse=True)
    def init_postgres_dsn(cls, values: dict[str, Any]) -> dict[str, Any]:
        return {
            **values,
            "dsn": (
                f"postgresql://{values['user']}:{values['password']}"
                f"@{values['host']}:{values['port']}/{values['db']}"
            ),
            "dsn_async": (
                f"postgresql+asyncpg://{values['user']}:{values['password']}"
                f"@{values['host']}:{values['port']}/{values['db']}"
            ),
        }

    class Config:
        env_prefix = "POSTGRES_"
        env_file = ".env"


class Settings(BaseSettings):
    paths: PathsSettings = PathsSettings()
    postgres: PostgresSettings = PostgresSettings()


settings = Settings()

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


if __name__ == "__main__":
    print(settings.postgres.postgres_dsn)

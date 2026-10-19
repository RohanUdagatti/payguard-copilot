"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for PayGuard Copilot.

    All values are read from environment variables. Secrets must never be
    hardcoded; use ``.env`` locally and a secret manager in production.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application
    app_name: str = Field(default="payguard-copilot")
    app_version: str = Field(default="0.1.0")
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # PostgreSQL (application database)
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_user: str = Field(default="payguard")
    postgres_password: str = Field(default="changeme")
    postgres_db: str = Field(default="payguard")

    # Redis
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_password: str = Field(default="")

    # Qdrant
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_api_key: str = Field(default="")

    # Langfuse
    langfuse_host: str = Field(default="http://localhost:3000")
    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: str = Field(default="")
    langfuse_db_user: str = Field(default="langfuse")
    langfuse_db_password: str = Field(default="changeme")
    langfuse_db_name: str = Field(default="langfuse")
    langfuse_nextauth_secret: str = Field(default="changeme-nextauth-secret")
    langfuse_salt: str = Field(default="changeme-salt")
    langfuse_encryption_key: str = Field(
        default="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    )
    langfuse_nextauth_url: str = Field(default="http://localhost:3000")

    @property
    def postgres_dsn(self) -> str:
        """Build a PostgreSQL connection URL from discrete settings."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        """Build a Redis URL from discrete settings."""
        if self.redis_password:
            return (
                f"redis://:{self.redis_password}"
                f"@{self.redis_host}:{self.redis_port}/0"
            )
        return f"redis://{self.redis_host}:{self.redis_port}/0"

    @property
    def qdrant_url(self) -> str:
        """Build a Qdrant HTTP base URL."""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()

"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field, field_validator
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
    qdrant_collection: str = Field(default="payguard_docs")

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

    # LLM / RAG
    llm_provider: str = Field(default="gemini")
    llm_model: str = Field(default="gemini-3.8-flash")
    llm_timeout_seconds: float = Field(default=60.0)
    llm_max_retries: int = Field(default=3)
    gemini_api_key: str = Field(default="")
    google_api_key: str = Field(default="")
    openai_api_key: str = Field(default="")

    embedding_provider: str = Field(default="local")
    embedding_model_local: str = Field(default="BAAI/bge-small-en-v1.5")
    embedding_model_openai: str = Field(default="text-embedding-3-small")
    embedding_model_gemini: str = Field(default="models/text-embedding-004")
    embedding_batch_size: int = Field(default=32)
    embedding_timeout_seconds: float = Field(default=60.0)

    rag_top_k: int = Field(default=5)
    rag_score_threshold: float = Field(default=0.2)
    rag_docs_dir: str = Field(default="data/raw_docs")

    @field_validator("llm_provider", "embedding_provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    def resolve_gemini_api_key(self) -> str:
        """Return Gemini key, preferring GEMINI_API_KEY over GOOGLE_API_KEY."""
        return self.gemini_api_key or self.google_api_key

    def require_active_llm_key(self) -> str:
        """Return the API key for the active LLM provider or raise ValueError."""
        provider = self.llm_provider
        if provider == "gemini":
            key = self.resolve_gemini_api_key()
            if not key:
                raise ValueError(
                    "LLM_PROVIDER=gemini requires GEMINI_API_KEY "
                    "(or GOOGLE_API_KEY as fallback) in the environment."
                )
            return key
        if provider == "openai":
            if not self.openai_api_key:
                raise ValueError(
                    "LLM_PROVIDER=openai requires OPENAI_API_KEY in the environment."
                )
            return self.openai_api_key
        raise ValueError(
            f"Unsupported LLM_PROVIDER={provider!r}. Supported: gemini, openai."
        )

    def require_active_embedding_key(self) -> str | None:
        """Validate the active embedding provider and return a key if required."""
        provider = self.embedding_provider
        if provider == "local":
            return None
        if provider == "openai":
            if not self.openai_api_key:
                raise ValueError(
                    "EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY "
                    "in the environment."
                )
            return self.openai_api_key
        if provider == "gemini":
            key = self.resolve_gemini_api_key()
            if not key:
                raise ValueError(
                    "EMBEDDING_PROVIDER=gemini requires GEMINI_API_KEY "
                    "(or GOOGLE_API_KEY as fallback) in the environment."
                )
            return key
        raise ValueError(
            f"Unsupported EMBEDDING_PROVIDER={provider!r}. "
            "Supported: local, openai, gemini."
        )

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

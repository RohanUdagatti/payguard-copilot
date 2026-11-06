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

    @field_validator("llm_model")
    @classmethod
    def _validate_llm_model(cls, v: str) -> str:
        if v and v.endswith("-latest"):
            raise ValueError(
                f"LLM_MODEL={v!r} is invalid: must be a pinned model name, "
                "not an unpinned '-latest' alias."
            )
        return v

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

    # Phase 3: Advanced Retrieval Feature Flags & Configs
    retrieval_mode: str = Field(default="dense")
    rerank_enabled: bool = Field(default=False)
    rerank_model: str = Field(default="BAAI/bge-reranker-base")
    rerank_top_n: int = Field(default=20)
    parent_child_enabled: bool = Field(default=False)
    child_chunk_tokens: int = Field(default=200)
    child_chunk_overlap: int = Field(default=50)
    multi_query_enabled: bool = Field(default=False)
    multi_query_count: int = Field(default=3)
    temporal_filter_enabled: bool = Field(default=False)
    cache_enabled: bool = Field(default=False)
    cache_similarity_threshold: float = Field(default=0.998)
    cache_ttl_seconds: int = Field(default=3600)
    ingest_version: str = Field(default="1.0.0")

    # Phase 4: Evaluation & Judge Settings
    judge_provider: str = Field(default="gemini")
    judge_model: str = Field(default="")
    judge_rpm: int = Field(default=5)
    judge_rpd: int = Field(default=20)
    judge_timeout_seconds: float = Field(default=60.0)
    eval_run_id: str = Field(default="")

    @field_validator(
        "llm_provider", "embedding_provider", "retrieval_mode", mode="before"
    )
    @classmethod
    def _normalize_provider(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    def retrieval_config_hash(self) -> str:
        """Return a deterministic hash of active retrieval settings."""
        import hashlib

        payload = (
            f"{self.retrieval_mode}:"
            f"{self.rerank_enabled}:{self.rerank_model}:{self.rerank_top_n}:"
            f"{self.parent_child_enabled}:{self.multi_query_enabled}:"
            f"{self.temporal_filter_enabled}:{self.rag_top_k}:"
            f"{self.embedding_provider}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

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

    def validate_judge_config(self) -> None:
        """Validate judge model configuration according to Phase 4 rules."""
        if not self.judge_model:
            raise ValueError(
                "JUDGE_MODEL has no default and must be explicitly set "
                "via environment or flag."
            )
        if "-latest" in self.judge_model:
            raise ValueError(
                f"JUDGE_MODEL={self.judge_model!r} is invalid: must be a pinned "
                "model name, not an unpinned '-latest' alias."
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

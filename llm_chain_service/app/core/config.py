from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENROUTER_API_KEY: str

    MODEL_STAGE1: str
    MODEL_STAGE2: str
    MODEL_STAGE3: str

    TIMEOUT_SEC: int = 30
    MAX_TOKENS: int = 2000
    RETRIES: int = 2

    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


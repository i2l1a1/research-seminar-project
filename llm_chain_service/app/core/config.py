from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openrouter_api_key: str = Field(validation_alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        validation_alias="OPENROUTER_BASE_URL",
    )

    model_step1: str = Field(validation_alias="MODEL_STEP1")
    model_step2: str = Field(validation_alias="MODEL_STEP2")
    model_step3: str = Field(validation_alias="MODEL_STEP3")
    model_step4: str = Field(validation_alias="MODEL_STEP4")
    model_step5: str = Field(validation_alias="MODEL_STEP5")

    timeout_sec: int = Field(default=30, validation_alias="TIMEOUT_SEC")
    max_tokens: int = Field(default=2000, validation_alias="MAX_TOKENS")
    retries: int = Field(default=2, validation_alias="RETRIES")

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )


settings = Settings()

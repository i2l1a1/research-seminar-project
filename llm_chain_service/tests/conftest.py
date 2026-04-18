"""Pytest: подгрузка `.env` из корня `llm_chain_service` до выполнения тестов."""

from pathlib import Path

from dotenv import load_dotenv

_SERVICE_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_SERVICE_ROOT / ".env")

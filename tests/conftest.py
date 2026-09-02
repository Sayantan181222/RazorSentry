import os
import sys

# Add project root to sys.path so src/ is importable in CI and locally
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("DATABASE_URL", "sqlite:///test_razorsentry.db")
os.environ.setdefault("MODEL_PATH", "models/lgbm_model.pkl")
os.environ.setdefault("THRESHOLD_PATH", "models/threshold.txt")

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.service import app, lifespan


# Provides an async test client wired directly to the FastAPI app
@pytest.fixture
async def client():
    async with lifespan(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac


# Returns True if the trained model artifact exists on disk
@pytest.fixture
def model_available() -> bool:
    return (
        os.path.exists("models/lgbm_model.pkl")
        and os.path.exists("models/threshold.txt")
    )

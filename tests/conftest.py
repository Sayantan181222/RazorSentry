import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.service import app, lifespan

# pyrefly: ignore [missing-import]
import pytest
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("DATABASE_URL", "sqlite:///test_razorsentry.db")

# Provides an async httpx client wired directly to the FastAPI app via ASGITransport
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

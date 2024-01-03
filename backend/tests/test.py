from fastapi import FastAPI
import pytest
from httpx import AsyncClient
from main import app
from PIL import Image
import io


@pytest.fixture
def test_app():
    return app


@pytest.mark.asyncio
async def test_predict_endpoint_with_valid_file(test_app: FastAPI):
    # メモリ上でシンプルな画像を生成
    image = Image.new('RGB', (100, 100), color='red')
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    image_bytes = buf.getvalue()

    async with AsyncClient(app=test_app, base_url="http://test") as ac:
        response = await ac.post(
            "/predict",
            files={"file": ("filename.jpg", image_bytes, "image/jpeg")}
        )
    assert response.status_code == 200
    assert "result" in response.json()
    assert "confidence" in response.json()


@pytest.mark.asyncio
async def test_predict_endpoint_with_invalid_file_type(test_app: FastAPI):
    async with AsyncClient(app=test_app, base_url="http://test") as ac:
        response = await ac.post(
            "/predict",
            files={"file": ("filename.txt", b"fake-text-data", "text/plain")}
        )
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]

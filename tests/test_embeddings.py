import pytest
import numpy as np
from pathlib import Path
from PIL import Image
import shutil
import asyncio

# Mock settings before importing app
import os
os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"
os.environ["CLIP_MODEL"] = "clip-ViT-B-32"
# Force CPU for tests to ensure they run everywhere and faster for simple check
os.environ["CUDA_VISIBLE_DEVICES"] = "" 

from app.services.embeddings import embedding_service

# Helper to create a dummy image
def create_dummy_image(path: Path):
    img = Image.new('RGB', (100, 100), color = 'red')
    img.save(path)

@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    return d

def test_embed_text_single():
    text = "Hello world"
    # Ensure service uses CPU as per env, or at least doesn't crash
    embedding = embedding_service.embed_text(text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] > 0
    # all-MiniLM-L6-v2 dim is 384
    assert embedding.shape[0] == 384

def test_embed_text_list():
    texts = ["Hello world", "Another sentence"]
    embeddings = embedding_service.embed_text(texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    assert isinstance(embeddings[0], np.ndarray)
    assert embeddings[0].shape[0] == 384

def test_embed_image(data_dir):
    img_path = data_dir / "test_image.jpg"
    create_dummy_image(img_path)
    
    embedding = embedding_service.embed_image(img_path)
    assert isinstance(embedding, np.ndarray)
    # CLIP vit-base-patch32 dim is 512
    assert embedding.shape[0] == 512

def test_embed_mixed_batch(data_dir):
    img_path = data_dir / "batch_image.jpg"
    create_dummy_image(img_path)
    
    items = ["Text item", img_path, "Another text"]
    embeddings = embedding_service.embed_batch(items)
    
    assert len(embeddings) == 3
    assert isinstance(embeddings[0], np.ndarray) # Text
    assert embeddings[0].shape[0] == 384
    assert isinstance(embeddings[1], np.ndarray) # Image
    assert embeddings[1].shape[0] == 512
    assert isinstance(embeddings[2], np.ndarray) # Text
    assert embeddings[2].shape[0] == 384

def test_caching():
    text = "Cached text"
    
    # First call
    emb1 = embedding_service.embed_text(text)
    
    # Verify it is in cache
    assert f"text:{text}" in embedding_service._cache
    
    # Modify cache manually to verify second call hits it
    fake_embedding = np.zeros(384)
    embedding_service._cache[f"text:{text}"] = fake_embedding
    
    # Second call
    emb2 = embedding_service.embed_text(text)
    
    # Should be the fake embedding
    assert np.array_equal(emb2, fake_embedding)

@pytest.mark.asyncio
async def test_async_wrappers(data_dir):
    img_path = data_dir / "async_image.jpg"
    create_dummy_image(img_path)
    
    # Text async
    text_emb = await embedding_service.embed_text_async("Async text")
    assert isinstance(text_emb, np.ndarray)
    assert text_emb.shape[0] == 384
    
    # Image async
    img_emb = await embedding_service.embed_image_async(img_path)
    assert isinstance(img_emb, np.ndarray)
    assert img_emb.shape[0] == 512
    
    # Batch async
    batch_emb = await embedding_service.embed_batch_async(["Async batch", img_path])
    assert len(batch_emb) == 2

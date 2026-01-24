"""
Unit tests for GeminiAgent service.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import io

from app.services.agent import GeminiAgent, get_agent
from google.api_core import exceptions as google_exceptions


@pytest.fixture
def agent():
    """Create a GeminiAgent instance for testing."""
    with patch('google.generativeai.configure'):
        with patch('google.generativeai.GenerativeModel'):
            agent = GeminiAgent(api_key="test_api_key", model="gemini-1.5-flash")
            return agent


@pytest.fixture
def mock_response():
    """Create a mock Gemini API response."""
    response = Mock()
    response.text = "This is a test summary response."
    return response


@pytest.fixture
def mock_streaming_response():
    """Create a mock streaming response."""
    chunks = [
        Mock(text="This is "),
        Mock(text="a streaming "),
        Mock(text="response.")
    ]
    return chunks


@pytest.fixture
def sample_image(tmp_path):
    """Create a sample test image."""
    img_path = tmp_path / "test_image.png"
    img = Image.new('RGB', (100, 100), color='red')
    img.save(img_path)
    return img_path


class TestGeminiAgentInitialization:
    """Test GeminiAgent initialization."""
    
    def test_init_default_model(self, agent):
        """Test initialization with default model."""
        assert agent.model_name == "gemini-1.5-flash"
        assert agent.api_key == "test_api_key"
    
    def test_init_custom_model(self):
        """Test initialization with custom model."""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel'):
                agent = GeminiAgent(api_key="test_key", model="gemini-1.5-pro")
                assert agent.model_name == "gemini-1.5-pro"


class TestGenerateSummary:
    """Test generate_summary method."""
    
    def test_text_only_generation(self, agent, mock_response):
        """Test basic text-only summary generation."""
        agent.model.generate_content = Mock(return_value=mock_response)
        
        result = agent.generate_summary(
            query="What is AI?",
            text_context=["AI is artificial intelligence."],
            visual_context=None,
            stream=False
        )
        
        assert result == "This is a test summary response."
        agent.model.generate_content.assert_called_once()
    
    def test_multimodal_generation(self, agent, mock_response, sample_image):
        """Test multimodal generation with images."""
        agent.model.generate_content = Mock(return_value=mock_response)
        
        result = agent.generate_summary(
            query="Describe the chart",
            text_context=["Figure 1 shows trends."],
            visual_context=[sample_image],
            stream=False
        )
        
        assert result == "This is a test summary response."
        # Should be called with prompt + images
        args = agent.model.generate_content.call_args[0][0]
        assert isinstance(args, list)
        assert len(args) > 1  # prompt + at least one image
    
    def test_streaming_generation(self, agent, mock_streaming_response):
        """Test streaming response generation."""
        agent.model.generate_content = Mock(return_value=iter(mock_streaming_response))
        
        result = agent.generate_summary(
            query="Summarize this",
            text_context=["Some context"],
            stream=True
        )
        
        # Collect all chunks
        chunks = list(result)
        assert len(chunks) == 3
        assert ''.join(chunks) == "This is a streaming response."
    
    def test_image_encoding_fallback(self, agent, mock_response):
        """Test fallback to text-only when image encoding fails."""
        agent.model.generate_content = Mock(return_value=mock_response)
        
        # Pass non-existent image path
        result = agent.generate_summary(
            query="Test query",
            text_context=["Context"],
            visual_context=[Path("/nonexistent/image.png")],
            stream=False
        )
        
        # Should still succeed with text-only
        assert result == "This is a test summary response."
    
    def test_empty_context(self, agent, mock_response):
        """Test generation with empty context."""
        agent.model.generate_content = Mock(return_value=mock_response)
        
        result = agent.generate_summary(
            query="Test query",
            text_context=[],
            visual_context=None,
            stream=False
        )
        
        assert result == "This is a test summary response."


class TestAsyncGeneration:
    """Test async generation methods."""
    
    @pytest.mark.asyncio
    async def test_async_generation(self, agent, mock_response):
        """Test async summary generation."""
        agent.model.generate_content = Mock(return_value=mock_response)
        
        result = await agent.generate_summary_async(
            query="Test query",
            text_context=["Context"],
            stream=False
        )
        
        assert result == "This is a test summary response."


class TestPromptBuilding:
    """Test prompt building functionality."""
    
    def test_build_prompt_basic(self, agent):
        """Test basic prompt building."""
        prompt = agent._build_prompt(
            query="What is machine learning?",
            text_context=["ML is a subset of AI."]
        )
        
        assert "What is machine learning?" in prompt
        assert "ML is a subset of AI." in prompt
        assert "Query:" in prompt
        assert "Text Context:" in prompt
    
    def test_build_prompt_multiple_contexts(self, agent):
        """Test prompt with multiple text contexts."""
        prompt = agent._build_prompt(
            query="Test query",
            text_context=["Context 1", "Context 2", "Context 3"]
        )
        
        assert "Context 1" in prompt
        assert "Context 2" in prompt
        assert "Context 3" in prompt
    
    def test_build_prompt_empty_context(self, agent):
        """Test prompt with empty context."""
        prompt = agent._build_prompt(
            query="Test query",
            text_context=[]
        )
        
        assert "Test query" in prompt
        assert "No text context available" in prompt


class TestImageEncoding:
    """Test image encoding functionality."""
    
    def test_encode_single_image(self, agent, sample_image):
        """Test encoding a single image."""
        images = agent._encode_images([sample_image])
        
        assert len(images) == 1
        assert isinstance(images[0], Image.Image)
        assert images[0].mode == 'RGB'
    
    def test_encode_multiple_images(self, agent, tmp_path):
        """Test encoding multiple images."""
        # Create multiple test images
        img_paths = []
        for i in range(3):
            img_path = tmp_path / f"test_{i}.png"
            img = Image.new('RGB', (50, 50), color='blue')
            img.save(img_path)
            img_paths.append(img_path)
        
        images = agent._encode_images(img_paths)
        
        assert len(images) == 3
        assert all(isinstance(img, Image.Image) for img in images)
    
    def test_encode_nonexistent_image(self, agent):
        """Test encoding with non-existent image path."""
        images = agent._encode_images([Path("/fake/image.png")])
        
        # Should return empty list (skip invalid images)
        assert len(images) == 0
    
    def test_encode_mixed_valid_invalid(self, agent, sample_image):
        """Test encoding with mix of valid and invalid images."""
        images = agent._encode_images([
            sample_image,
            Path("/fake/image.png"),
            sample_image
        ])
        
        # Should only encode valid images
        assert len(images) == 2


class TestRetryLogic:
    """Test retry logic with exponential backoff."""
    
    def test_successful_first_attempt(self, agent):
        """Test successful execution on first attempt."""
        func = Mock(return_value="Success")
        
        result = agent._retry_with_backoff(func)
        
        assert result == "Success"
        assert func.call_count == 1
    
    def test_retry_on_rate_limit(self, agent):
        """Test retry on rate limit error."""
        func = Mock(side_effect=[
            google_exceptions.ResourceExhausted("Rate limited"),
            google_exceptions.ResourceExhausted("Rate limited"),
            "Success"
        ])
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = agent._retry_with_backoff(func, max_retries=3)
        
        assert result == "Success"
        assert func.call_count == 3
    
    def test_retry_exhausted(self, agent):
        """Test when all retries are exhausted."""
        func = Mock(side_effect=google_exceptions.ResourceExhausted("Rate limited"))
        
        with patch('time.sleep'):
            with pytest.raises(google_exceptions.ResourceExhausted):
                agent._retry_with_backoff(func, max_retries=2)
        
        assert func.call_count == 3  # initial + 2 retries
    
    def test_no_retry_on_invalid_argument(self, agent):
        """Test that invalid arguments don't trigger retry."""
        func = Mock(side_effect=google_exceptions.InvalidArgument("Bad input"))
        
        with pytest.raises(google_exceptions.InvalidArgument):
            agent._retry_with_backoff(func, max_retries=3)
        
        # Should fail immediately without retry
        assert func.call_count == 1
    
    def test_retry_on_server_error(self, agent):
        """Test retry on server error."""
        func = Mock(side_effect=[
            google_exceptions.InternalServerError("Server error"),
            "Success"
        ])
        
        with patch('time.sleep'):
            result = agent._retry_with_backoff(func, max_retries=3)
        
        assert result == "Success"
        assert func.call_count == 2


class TestPageReferenceExtraction:
    """Test page reference extraction."""
    
    def test_extract_page_pattern_1(self, agent):
        """Test extraction of 'Page N' pattern."""
        context = ["This is on Page 5", "See Page 10 for details"]
        pages = agent._extract_page_references(context)
        
        assert 5 in pages
        assert 10 in pages
    
    def test_extract_page_pattern_2(self, agent):
        """Test extraction of 'p. N' pattern."""
        context = ["Refer to p. 3", "As shown in p.7"]
        pages = agent._extract_page_references(context)
        
        assert 3 in pages
        assert 7 in pages
    
    def test_no_duplicates(self, agent):
        """Test that duplicate page numbers are removed."""
        context = ["Page 5", "page 5", "p. 5"]
        pages = agent._extract_page_references(context)
        
        assert pages.count(5) == 1
    
    def test_sorted_output(self, agent):
        """Test that page numbers are sorted."""
        context = ["Page 10", "Page 3", "Page 7"]
        pages = agent._extract_page_references(context)
        
        assert pages == [3, 7, 10]


class TestSingletonInstance:
    """Test singleton instance functionality."""
    
    def test_get_agent(self):
        """Test get_agent returns instance."""
        with patch('app.services.agent.settings') as mock_settings:
            mock_settings.GOOGLE_API_KEY = "test_key"
            mock_settings.GEMINI_MODEL = "gemini-1.5-flash"
            
            with patch('google.generativeai.configure'):
                with patch('google.generativeai.GenerativeModel'):
                    agent = get_agent()
                    assert agent is not None
                    assert isinstance(agent, GeminiAgent)
    
    def test_get_agent_no_api_key(self):
        """Test get_agent raises error without API key."""
        with patch('app.services.agent.settings') as mock_settings:
            mock_settings.GOOGLE_API_KEY = None
            
            # Reset singleton
            import app.services.agent
            app.services.agent.gemini_agent = None
            
            with pytest.raises(ValueError, match="GOOGLE_API_KEY not configured"):
                get_agent()

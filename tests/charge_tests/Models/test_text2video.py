import os

import pytest

import lazyllm
from lazyllm.components.formatter import decode_query_with_filepaths

from ...utils import get_api_key, get_path


BASE_PATH = 'lazyllm/module/llms/onlinemodule/base/onlineMultiModalBase.py'


class TestText2Video:
    """Test cases for text-to-video and image-to-video generation modules."""

    test_video_prompt = '一只可爱的猫咪在草地上奔跑'
    test_image_prompt = '让这只猫向前走'

    @staticmethod
    def _check_video_result(result):
        """Validate that result contains valid video file paths."""
        assert result is not None
        assert isinstance(result, str)
        assert result.startswith('<lazyllm-query>')

        decoded = decode_query_with_filepaths(result)
        assert 'files' in decoded
        assert len(decoded['files']) > 0

        file = decoded['files'][0]
        assert os.path.exists(file)
        assert file.endswith(('.mp4', '.webm', '.avi', '.mov'))

    def common_text2video(self, source, type='text2video', **kwargs):
        """Common test method for text-to-video generation."""
        api_key = get_api_key(source)
        t2v = lazyllm.OnlineMultiModalModule(source=source, type=type, api_key=api_key, **kwargs)
        result = t2v(self.test_video_prompt)
        self._check_video_result(result)
        return result

    def common_image2video(self, source, image_path, type='image2video', **kwargs):
        """Common test method for image-to-video generation."""
        api_key = get_api_key(source)
        i2v = lazyllm.OnlineMultiModalModule(source=source, type=type, api_key=api_key, **kwargs)
        result = i2v(self.test_image_prompt, lazyllm_files=[image_path])
        self._check_video_result(result)
        return result

    # ==================== Qwen Text2Video Tests ====================

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('qwen'))
    @pytest.mark.xfail(reason='Video generation API may have rate limits or require special permissions')
    def test_qwen_text2video(self):
        """Test Qwen text-to-video generation with wanx2.1-t2v-turbo model."""
        self.common_text2video(source='qwen', type='text2video')

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('qwen'))
    @pytest.mark.xfail(reason='Video generation API may have rate limits or require special permissions')
    def test_qwen_text2video_plus(self):
        """Test Qwen text-to-video generation with wanx2.1-t2v-plus model."""
        self.common_text2video(source='qwen', type='text2video', model='wanx2.1-t2v-plus')

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('qwen'))
    @pytest.mark.xfail(reason='Video generation API may have rate limits or require special permissions')
    def test_qwen_text2video_custom_params(self):
        """Test Qwen text-to-video with custom parameters."""
        api_key = get_api_key('qwen')
        t2v = lazyllm.OnlineMultiModalModule(
            source='qwen',
            type='text2video',
            api_key=api_key,
            model='wanx2.1-t2v-turbo'
        )
        result = t2v(
            self.test_video_prompt,
            size='720*1280',  # Vertical video
            duration=5,
            prompt_extend=True
        )
        self._check_video_result(result)

    # ==================== Qwen Image2Video Tests ====================

    @pytest.fixture
    def sample_image_path(self, tmp_path):
        """Create a sample image for testing."""
        from PIL import Image
        img_path = tmp_path / 'test_image.png'
        img = Image.new('RGB', (512, 512), color='blue')
        img.save(img_path)
        return str(img_path)

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('qwen'))
    @pytest.mark.xfail(reason='Video generation API may have rate limits or require special permissions')
    def test_qwen_image2video(self, sample_image_path):
        """Test Qwen image-to-video generation with wanx2.1-i2v-turbo model."""
        self.common_image2video(source='qwen', image_path=sample_image_path, type='image2video')

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('qwen'))
    @pytest.mark.xfail(reason='Video generation API may have rate limits or require special permissions')
    def test_qwen_image2video_plus(self, sample_image_path):
        """Test Qwen image-to-video generation with wanx2.1-i2v-plus model."""
        self.common_image2video(
            source='qwen',
            image_path=sample_image_path,
            type='image2video',
            model='wanx2.1-i2v-plus'
        )

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('qwen'))
    @pytest.mark.xfail(reason='Image2Video requires image file')
    def test_qwen_image2video_without_image(self):
        """Test that image2video fails without image file."""
        api_key = get_api_key('qwen')
        i2v = lazyllm.OnlineMultiModalModule(source='qwen', type='image2video', api_key=api_key)
        with pytest.raises(ValueError, match='requires at least one image file'):
            i2v(self.test_image_prompt)

    # ==================== Doubao Text2Video Tests ====================

    @pytest.mark.ignore_cache_on_change(get_path('doubao'))
    @pytest.mark.xfail(reason='Video generation API may have rate limits or require special permissions')
    def test_doubao_text2video(self):
        """Test Doubao text-to-video generation."""
        self.common_text2video(source='doubao', type='text2video')

    @pytest.mark.ignore_cache_on_change(get_path('doubao'))
    @pytest.mark.xfail(reason='Video generation API may have rate limits or require special permissions')
    def test_doubao_text2video_custom_params(self):
        """Test Doubao text-to-video with custom parameters."""
        api_key = get_api_key('doubao')
        t2v = lazyllm.OnlineMultiModalModule(
            source='doubao',
            type='text2video',
            api_key=api_key
        )
        result = t2v(
            self.test_video_prompt,
            size='720x1280',
            duration=5
        )
        self._check_video_result(result)

    # ==================== Doubao Image2Video Tests ====================

    @pytest.mark.ignore_cache_on_change(get_path('doubao'))
    @pytest.mark.xfail(reason='Video generation API may have rate limits or require special permissions')
    def test_doubao_image2video(self, sample_image_path):
        """Test Doubao image-to-video generation."""
        self.common_image2video(source='doubao', image_path=sample_image_path, type='image2video')

    @pytest.mark.ignore_cache_on_change(get_path('doubao'))
    @pytest.mark.xfail(reason='Image2Video requires image file')
    def test_doubao_image2video_without_image(self):
        """Test that image2video fails without image file."""
        api_key = get_api_key('doubao')
        i2v = lazyllm.OnlineMultiModalModule(source='doubao', type='image2video', api_key=api_key)
        with pytest.raises(ValueError, match='requires at least one image file'):
            i2v(self.test_image_prompt)

    # ==================== Factory Tests ====================

    def test_factory_text2video_type(self):
        """Test that factory correctly creates text2video module type."""
        # Test with explicit type
        module = lazyllm.OnlineMultiModalModule(source='qwen', type='text2video', skip_auth=True,
                                                 base_url='http://localhost:8000')
        assert module is not None

    def test_factory_image2video_type(self):
        """Test that factory correctly creates image2video module type."""
        # Test with explicit type
        module = lazyllm.OnlineMultiModalModule(source='qwen', type='image2video', skip_auth=True,
                                                 base_url='http://localhost:8000')
        assert module is not None

    def test_factory_function_alias(self):
        """Test that function parameter works as alias for type."""
        module1 = lazyllm.OnlineMultiModalModule(source='qwen', function='text2video', skip_auth=True,
                                                  base_url='http://localhost:8000')
        module2 = lazyllm.OnlineMultiModalModule(source='qwen', type='text2video', skip_auth=True,
                                                  base_url='http://localhost:8000')
        assert type(module1) == type(module2)


class TestLLMTypeEnumExtension:
    """Test cases for LLMType enum extension."""

    def test_text2video_type_exists(self):
        """Test that TEXT2VIDEO type exists in LLMType enum."""
        from lazyllm.components.utils.downloader.model_downloader import LLMType
        assert hasattr(LLMType, 'TEXT2VIDEO')
        assert LLMType.TEXT2VIDEO.value == 'TEXT2VIDEO'

    def test_image2video_type_exists(self):
        """Test that IMAGE2VIDEO type exists in LLMType enum."""
        from lazyllm.components.utils.downloader.model_downloader import LLMType
        assert hasattr(LLMType, 'IMAGE2VIDEO')
        assert LLMType.IMAGE2VIDEO.value == 'IMAGE2VIDEO'

    def test_type_case_insensitive(self):
        """Test that LLMType lookup is case-insensitive."""
        from lazyllm.components.utils.downloader.model_downloader import LLMType
        assert LLMType['text2video'] == LLMType.TEXT2VIDEO
        assert LLMType['TEXT2VIDEO'] == LLMType.TEXT2VIDEO
        assert LLMType['Text2Video'] == LLMType.TEXT2VIDEO

    def test_type_normalization(self):
        """Test that type normalization works correctly."""
        from lazyllm.components.utils.downloader.model_downloader import LLMType
        assert LLMType._normalize('text2video') == 'text2video'
        assert LLMType._normalize('TEXT2VIDEO') == 'text2video'

import lazyllm
from typing import Dict, List, Union
from lazyllm.components.utils.downloader.model_downloader import LLMType
from ..base import (
    OnlineChatModuleBase, LazyLLMOnlineEmbedModuleBase,
    LazyLLMOnlineMultimodalEmbedModuleBase, LazyLLMOnlineText2ImageModuleBase,
    LazyLLMOnlineText2VideoModuleBase, LazyLLMOnlineImage2VideoModuleBase
)
import requests
from lazyllm.components.formatter import encode_query_with_filepaths
from lazyllm.components.utils.file_operate import bytes_to_file
from lazyllm.thirdparty import volcenginesdkarkruntime
from lazyllm import LOG


class DoubaoChat(OnlineChatModuleBase):
    MODEL_NAME = 'doubao-1-5-pro-32k-250115'
    VLM_MODEL_PREFIX = ['doubao-seed-1-6-vision', 'doubao-1-5-ui-tars']

    def __init__(self, model: str = None, base_url: str = 'https://ark.cn-beijing.volces.com/api/v3/',
                 api_key: str = None, stream: bool = True, return_trace: bool = False, **kwargs):
        super().__init__(api_key=api_key or lazyllm.config['doubao_api_key'], base_url=base_url,
                         model_name=model or lazyllm.config['doubao_model_name'] or DoubaoChat.MODEL_NAME,
                         stream=stream, return_trace=return_trace, **kwargs)

    def _get_system_prompt(self):
        return ('You are Doubao, an AI assistant. Your task is to provide appropriate responses '
                'and support to user\'s questions and requests.')

    def _validate_api_key(self):
        '''Validate API Key by sending a minimal request'''
        try:
            # Doubao (Volcano Engine) validates API key using a minimal chat request
            data = {
                'model': self._model_name,
                'messages': [{'role': 'user', 'content': 'hi'}],
                'max_tokens': 1  # Only generate 1 token for validation
            }
            response = requests.post(self._chat_url, headers=self._header, json=data, timeout=10)
            return response.status_code == 200
        except Exception:
            return False


class DoubaoEmbed(LazyLLMOnlineEmbedModuleBase):
    def __init__(self,
                 embed_url: str = 'https://ark.cn-beijing.volces.com/api/v3/embeddings',
                 embed_model_name: str = 'doubao-embedding-text-240715',
                 api_key: str = None,
                 batch_size: int = 16,
                 **kw):
        super().__init__(embed_url, api_key or lazyllm.config['doubao_api_key'], embed_model_name,
                         batch_size=batch_size, **kw)


class DoubaoMultimodalEmbed(LazyLLMOnlineMultimodalEmbedModuleBase):
    def __init__(self,
                 embed_url: str = 'https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal',
                 embed_model_name: str = None,
                 api_key: str = None):
        embed_model_name = (embed_model_name or lazyllm.config['doubao_multimodal_embed_model_name']
                            or 'doubao-embedding-vision-241215')
        super().__init__(embed_url, api_key or lazyllm.config['doubao_api_key'], embed_model_name)

    def _encapsulated_data(self, input: Union[List, str], **kwargs) -> Dict[str, str]:
        if isinstance(input, str):
            input = [{'text': input}]
        elif isinstance(input, list):
            # Validate input format, at most 1 text segment + 1 image
            if len(input) == 0:
                raise ValueError('Input list cannot be empty')
            if len(input) > 2:
                raise ValueError('Input list must contain at most 2 items (1 text and/or 1 image)')
        else:
            raise ValueError('Input must be either a string or a list of dictionaries')

        json_data = {
            'input': input,
            'model': self._embed_model_name
        }
        if len(kwargs) > 0:
            json_data.update(kwargs)

        return json_data

    def _parse_response(self, response: Dict, input: Union[List, str]) -> List[float]:
        # Doubao multimodal embedding returns a single fused embedding
        return response['data']['embedding']


class DoubaoMultiModal():
    def __init__(self, api_key: str = None, url: str = ''):
        api_key = api_key or lazyllm.config['doubao_api_key']
        self._client = volcenginesdkarkruntime.Ark(base_url=url, api_key=api_key)


class DoubaoText2Image(LazyLLMOnlineText2ImageModuleBase, DoubaoMultiModal):
    MODEL_NAME = 'doubao-seedream-3-0-t2i-250415'
    IMAGE_EDITING_MODEL_NAME = 'doubao-seedream-3-0-t2i-250415'

    def __init__(self, api_key: str = None, model: str = None, url='https://ark.cn-beijing.volces.com/api/v3',
                 return_trace: bool = False, **kwargs):
        resolved_model = model or lazyllm.config['doubao_text2image_model_name'] or DoubaoText2Image.MODEL_NAME
        super().__init__(model=resolved_model, api_key=api_key, return_trace=return_trace, url=url, **kwargs)
        DoubaoMultiModal.__init__(self, api_key=api_key, url=url)

    def _forward(self, input: str = None, files: List[str] = None, n: int = 1, size: str = '1024x1024', seed: int = -1,
                 guidance_scale: float = 2.5, watermark: bool = True, model: str = None, url: str = None, **kwargs):
        has_ref_image = files is not None and len(files) > 0
        if self._type == LLMType.IMAGE_EDITING and not has_ref_image:
            LOG.warning(
                f'Image editing is enabled for model {self._model_name}, but no image file was provided. '
                f'Please provide an image file via the "files" parameter.'
            )
        if self._type != LLMType.IMAGE_EDITING and has_ref_image:
            msg = str(f'Image file was provided, but image editing is not enabled for model {self._model_name}. Please '
                      f'use default image-editing model {self.IMAGE_EDITING_MODEL_NAME} or other image-editing model.')
            raise ValueError(msg)

        if has_ref_image:
            image_results = self._load_images(files)
            contents = [f'data:image/png;base64,{base64_str}' for base64_str, _ in image_results]
        api_params = {
            'model': model,
            'prompt': input,
            'size': size,
            'seed': seed,
            'guidance_scale': guidance_scale,
            'watermark': watermark,
            **kwargs
        }
        if has_ref_image:
            api_params['image'] = contents
            if n > 1:
                api_params['sequential_image_generation'] = 'auto'
                max_images = min(n, 15)
                sigo = volcenginesdkarkruntime.types.images.SequentialImageGenerationOptions
                api_params['sequential_image_generation_options'] = sigo(max_images=max_images)
        imagesResponse = self._client.images.generate(**api_params)
        image_contents = [requests.get(result.url).content for result in imagesResponse.data]
        return encode_query_with_filepaths(None, bytes_to_file(image_contents))


class DoubaoText2Video(LazyLLMOnlineText2VideoModuleBase, DoubaoMultiModal):
    """
    Doubao Text-to-Video generation module using Volcano Engine video synthesis API.

    This module supports generating videos from text prompts using ByteDance's
    Doubao video generation models.

    Example:
        >>> t2v = DoubaoText2Video(model='doubao-video-generation')
        >>> result = t2v('A cat playing with a ball in the garden')
    """
    MODEL_NAME = 'doubao-seaweed-241128'

    def __init__(self, api_key: str = None, model: str = None,
                 url: str = 'https://ark.cn-beijing.volces.com/api/v3',
                 return_trace: bool = False, **kwargs):
        resolved_model = model or lazyllm.config.get('doubao_text2video_model_name', None) or DoubaoText2Video.MODEL_NAME
        super().__init__(model=resolved_model, api_key=api_key, return_trace=return_trace, url=url, **kwargs)
        DoubaoMultiModal.__init__(self, api_key=api_key, url=url)

    def _forward(self, input: str = None, size: str = '1280x720', duration: int = 5,
                 seed: int = None, url: str = None, model: str = None, **kwargs):
        """
        Generate video from text prompt.

        Args:
            input: Text prompt describing the video to generate
            size: Video resolution (e.g., '1280x720', '720x1280')
            duration: Video duration in seconds (default: 5)
            seed: Random seed for reproducibility
            url: Override base URL
            model: Override model name
            **kwargs: Additional parameters passed to the API

        Returns:
            Encoded query string with file paths to generated video files
        """
        api_params = {
            'model': model,
            'content': [
                {
                    'type': 'text',
                    'text': input
                }
            ]
        }

        # Add optional parameters
        if size:
            api_params['size'] = size
        if duration:
            api_params['duration'] = duration
        if seed is not None:
            api_params['seed'] = seed

        api_params.update(kwargs)

        try:
            # Use content_generation for video tasks
            response = self._client.content_generation.tasks.create(**api_params)
            task_id = response.id

            # Poll for completion
            import time
            max_wait = 600  # Maximum wait time in seconds
            poll_interval = 5
            elapsed = 0

            while elapsed < max_wait:
                task_status = self._client.content_generation.tasks.retrieve(task_id)
                if task_status.status == 'succeeded':
                    # Extract video URL
                    video_url = self._extract_video_url(task_status)
                    if not video_url:
                        raise Exception('No video URL found in response')
                    video_content = requests.get(video_url, timeout=300).content
                    return encode_query_with_filepaths(None, bytes_to_file(video_content, suffix='.mp4'))
                elif task_status.status == 'failed':
                    error_msg = getattr(task_status, 'error', {}).get('message', 'Unknown error')
                    raise Exception(f'Video generation failed: {error_msg}')
                time.sleep(poll_interval)
                elapsed += poll_interval

            raise Exception(f'Video generation timed out after {max_wait} seconds')

        except AttributeError:
            # Fallback: Try using bot.chat interface for video generation
            return self._forward_via_chat(input, size, duration, seed, model, **kwargs)

    def _forward_via_chat(self, input: str, size: str, duration: int,
                          seed: int, model: str, **kwargs):
        """Fallback method using chat interface for video generation."""
        messages = [
            {
                'role': 'user',
                'content': input
            }
        ]

        api_params = {
            'model': model,
            'messages': messages,
        }

        if size:
            api_params['size'] = size
        if duration:
            api_params['duration'] = duration
        if seed is not None:
            api_params['seed'] = seed

        api_params.update(kwargs)

        response = self._client.chat.completions.create(**api_params)

        # Extract video URL from response
        video_url = self._extract_video_url_from_chat(response)
        if not video_url:
            raise Exception('No video URL found in response')

        video_content = requests.get(video_url, timeout=300).content
        return encode_query_with_filepaths(None, bytes_to_file(video_content, suffix='.mp4'))

    def _extract_video_url(self, response):
        """Extract video URL from task response."""
        try:
            if hasattr(response, 'output') and response.output:
                if hasattr(response.output, 'video_url'):
                    return response.output.video_url
                if hasattr(response.output, 'results') and response.output.results:
                    return response.output.results[0].url
            if hasattr(response, 'data') and response.data:
                if isinstance(response.data, list) and len(response.data) > 0:
                    return getattr(response.data[0], 'url', None)
            return None
        except Exception as e:
            LOG.error(f'Failed to extract video URL: {str(e)}')
            return None

    def _extract_video_url_from_chat(self, response):
        """Extract video URL from chat response."""
        try:
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and 'video_url' in item:
                            return item['video_url']
                        if hasattr(item, 'video_url'):
                            return item.video_url
                elif isinstance(content, str) and content.startswith('http'):
                    return content
            return None
        except Exception as e:
            LOG.error(f'Failed to extract video URL from chat: {str(e)}')
            return None


class DoubaoImage2Video(LazyLLMOnlineImage2VideoModuleBase, DoubaoMultiModal):
    """
    Doubao Image-to-Video generation module using Volcano Engine video synthesis API.

    This module supports generating videos from images with text prompts using
    ByteDance's Doubao video generation models.

    Example:
        >>> i2v = DoubaoImage2Video(model='doubao-video-generation')
        >>> result = i2v('Make the cat walk forward', lazyllm_files=['cat.jpg'])
    """
    MODEL_NAME = 'doubao-seaweed-241128'

    def __init__(self, api_key: str = None, model: str = None,
                 url: str = 'https://ark.cn-beijing.volces.com/api/v3',
                 return_trace: bool = False, **kwargs):
        resolved_model = model or lazyllm.config.get('doubao_image2video_model_name', None) or DoubaoImage2Video.MODEL_NAME
        super().__init__(model=resolved_model, api_key=api_key, return_trace=return_trace, url=url, **kwargs)
        DoubaoMultiModal.__init__(self, api_key=api_key, url=url)

    def _forward(self, input: str = None, files: List[str] = None, size: str = '1280x720',
                 duration: int = 5, seed: int = None, url: str = None, model: str = None, **kwargs):
        """
        Generate video from image with optional text prompt.

        Args:
            input: Text prompt describing the video motion/action (optional)
            files: List of image file paths or URLs (required, first image used)
            size: Video resolution (e.g., '1280x720', '720x1280')
            duration: Video duration in seconds (default: 5)
            seed: Random seed for reproducibility
            url: Override base URL
            model: Override model name
            **kwargs: Additional parameters passed to the API

        Returns:
            Encoded query string with file paths to generated video files
        """
        if not files or len(files) == 0:
            raise ValueError('Image2Video requires at least one image file. '
                             'Please provide image file(s) via the "files" parameter.')

        # Load and encode the first image
        image_results = self._load_images(files[:1])
        base64_str, _ = image_results[0]
        image_data = f'data:image/png;base64,{base64_str}'

        content = [
            {
                'type': 'image_url',
                'image_url': {
                    'url': image_data
                }
            }
        ]

        if input:
            content.append({
                'type': 'text',
                'text': input
            })

        api_params = {
            'model': model,
            'content': content
        }

        if size:
            api_params['size'] = size
        if duration:
            api_params['duration'] = duration
        if seed is not None:
            api_params['seed'] = seed

        api_params.update(kwargs)

        try:
            # Use content_generation for video tasks
            response = self._client.content_generation.tasks.create(**api_params)
            task_id = response.id

            # Poll for completion
            import time
            max_wait = 600  # Maximum wait time in seconds
            poll_interval = 5
            elapsed = 0

            while elapsed < max_wait:
                task_status = self._client.content_generation.tasks.retrieve(task_id)
                if task_status.status == 'succeeded':
                    video_url = self._extract_video_url(task_status)
                    if not video_url:
                        raise Exception('No video URL found in response')
                    video_content = requests.get(video_url, timeout=300).content
                    return encode_query_with_filepaths(None, bytes_to_file(video_content, suffix='.mp4'))
                elif task_status.status == 'failed':
                    error_msg = getattr(task_status, 'error', {}).get('message', 'Unknown error')
                    raise Exception(f'Video generation failed: {error_msg}')
                time.sleep(poll_interval)
                elapsed += poll_interval

            raise Exception(f'Video generation timed out after {max_wait} seconds')

        except AttributeError:
            # Fallback: Try using chat interface
            return self._forward_via_chat(input, image_data, size, duration, seed, model, **kwargs)

    def _forward_via_chat(self, input: str, image_data: str, size: str,
                          duration: int, seed: int, model: str, **kwargs):
        """Fallback method using chat interface for video generation."""
        content = [
            {
                'type': 'image_url',
                'image_url': {
                    'url': image_data
                }
            }
        ]
        if input:
            content.append({
                'type': 'text',
                'text': input
            })

        messages = [
            {
                'role': 'user',
                'content': content
            }
        ]

        api_params = {
            'model': model,
            'messages': messages,
        }

        if size:
            api_params['size'] = size
        if duration:
            api_params['duration'] = duration
        if seed is not None:
            api_params['seed'] = seed

        api_params.update(kwargs)

        response = self._client.chat.completions.create(**api_params)

        video_url = self._extract_video_url_from_chat(response)
        if not video_url:
            raise Exception('No video URL found in response')

        video_content = requests.get(video_url, timeout=300).content
        return encode_query_with_filepaths(None, bytes_to_file(video_content, suffix='.mp4'))

    def _extract_video_url(self, response):
        """Extract video URL from task response."""
        try:
            if hasattr(response, 'output') and response.output:
                if hasattr(response.output, 'video_url'):
                    return response.output.video_url
                if hasattr(response.output, 'results') and response.output.results:
                    return response.output.results[0].url
            if hasattr(response, 'data') and response.data:
                if isinstance(response.data, list) and len(response.data) > 0:
                    return getattr(response.data[0], 'url', None)
            return None
        except Exception as e:
            LOG.error(f'Failed to extract video URL: {str(e)}')
            return None

    def _extract_video_url_from_chat(self, response):
        """Extract video URL from chat response."""
        try:
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and 'video_url' in item:
                            return item['video_url']
                        if hasattr(item, 'video_url'):
                            return item.video_url
                elif isinstance(content, str) and content.startswith('http'):
                    return content
            return None
        except Exception as e:
            LOG.error(f'Failed to extract video URL from chat: {str(e)}')
            return None

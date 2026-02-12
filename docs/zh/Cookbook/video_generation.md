# 视频生成 (Video Generation)

LazyLLM 支持通过统一的接口调用多个供应商的视频生成 API，包括阿里万象 (Qwen) 和字节火山引擎 (Doubao)。本文档介绍如何使用视频生成功能。

## 目录

- [功能概述](#功能概述)
- [支持的模型](#支持的模型)
- [环境配置](#环境配置)
- [文本生成视频 (Text2Video)](#文本生成视频-text2video)
- [图片生成视频 (Image2Video)](#图片生成视频-image2video)
- [参数说明](#参数说明)
- [最佳实践](#最佳实践)

## 功能概述

视频生成功能支持两种模式：

1. **文本生成视频 (Text2Video)**: 根据文本提示词生成视频
2. **图片生成视频 (Image2Video)**: 根据输入图片和可选的文本提示词生成视频

## 支持的模型

### 阿里万象 (Qwen)

| 模型名称 | 类型 | 说明 |
|---------|------|------|
| `wanx2.1-t2v-turbo` | Text2Video | 快速文本生成视频模型 |
| `wanx2.1-t2v-plus` | Text2Video | 高质量文本生成视频模型 |
| `wanx2.1-i2v-turbo` | Image2Video | 快速图片生成视频模型 |
| `wanx2.1-i2v-plus` | Image2Video | 高质量图片生成视频模型 |

### 字节火山引擎 (Doubao)

| 模型名称 | 类型 | 说明 |
|---------|------|------|
| `doubao-seaweed-241128` | Text2Video/Image2Video | 豆包视频生成模型 |

## 环境配置

### 设置 API Key

```bash
# 阿里万象
export LAZYLLM_QWEN_API_KEY="your-qwen-api-key"

# 字节火山引擎
export LAZYLLM_DOUBAO_API_KEY="your-doubao-api-key"
```

### 安装依赖

```bash
# 阿里万象 SDK
pip install dashscope

# 字节火山引擎 SDK
pip install volcenginesdkarkruntime
```

## 文本生成视频 (Text2Video)

### 基本用法

```python
import lazyllm

# 使用阿里万象
t2v_qwen = lazyllm.OnlineMultiModalModule(
    source='qwen',
    type='text2video'
)

# 生成视频
result = t2v_qwen('一只可爱的猫咪在草地上奔跑')
print(result)  # 输出视频文件路径
```

### 使用字节火山引擎

```python
import lazyllm

# 使用字节火山引擎
t2v_doubao = lazyllm.OnlineMultiModalModule(
    source='doubao',
    type='text2video'
)

result = t2v_doubao('一只可爱的猫咪在草地上奔跑')
print(result)
```

### 指定模型和参数

```python
import lazyllm

# 使用高质量模型
t2v = lazyllm.OnlineMultiModalModule(
    source='qwen',
    type='text2video',
    model='wanx2.1-t2v-plus'
)

# 自定义参数
result = t2v(
    '一只可爱的猫咪在草地上奔跑',
    size='1280*720',      # 视频分辨率
    duration=5,           # 视频时长（秒）
    prompt_extend=True,   # 是否优化提示词
    seed=42               # 随机种子，用于复现
)
```

## 图片生成视频 (Image2Video)

### 基本用法

```python
import lazyllm

# 使用阿里万象
i2v_qwen = lazyllm.OnlineMultiModalModule(
    source='qwen',
    type='image2video'
)

# 从图片生成视频
result = i2v_qwen(
    '让这只猫向前走',           # 动作描述（可选）
    lazyllm_files=['cat.jpg']  # 输入图片
)
print(result)
```

### 使用字节火山引擎

```python
import lazyllm

# 使用字节火山引擎
i2v_doubao = lazyllm.OnlineMultiModalModule(
    source='doubao',
    type='image2video'
)

result = i2v_doubao(
    '让这只猫向前走',
    lazyllm_files=['cat.jpg']
)
print(result)
```

### 仅使用图片（无文本提示）

```python
import lazyllm

i2v = lazyllm.OnlineMultiModalModule(
    source='qwen',
    type='image2video',
    model='wanx2.1-i2v-turbo'
)

# 不提供文本描述，模型会自动推断动作
result = i2v(None, lazyllm_files=['landscape.jpg'])
```

## 参数说明

### 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `source` | str | - | 供应商名称：`qwen` 或 `doubao` |
| `type` | str | - | 功能类型：`text2video` 或 `image2video` |
| `model` | str | 默认模型 | 指定模型名称 |
| `api_key` | str | 环境变量 | API 密钥 |

### 视频生成参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `size` | str | `1280*720` | 视频分辨率，支持 `1280*720`, `720*1280`, `960*960` 等 |
| `duration` | int | 5 | 视频时长（秒） |
| `prompt_extend` | bool | True | 是否自动优化提示词 |
| `seed` | int | None | 随机种子，用于复现结果 |

### 注意事项

- **分辨率格式**：阿里万象使用 `*` 分隔（如 `1280*720`），字节火山引擎使用 `x` 分隔（如 `1280x720`）
- **视频时长**：不同模型支持的最大时长不同，请参考各供应商文档
- **图片要求**：Image2Video 至少需要一张输入图片

## 最佳实践

### 1. 编写高质量提示词

```python
# ✅ 好的提示词：具体、清晰、有动作描述
prompt = '一只橘色的猫咪在阳光明媚的花园里追逐蝴蝶，镜头跟随猫咪移动'

# ❌ 不好的提示词：太简单、缺乏细节
prompt = '猫'
```

### 2. 选择合适的分辨率

```python
# 横向视频（适合大屏幕播放）
size = '1280*720'

# 竖向视频（适合手机端、短视频平台）
size = '720*1280'

# 正方形视频（适合社交媒体）
size = '960*960'
```

### 3. 使用 seed 复现结果

```python
# 使用固定 seed 可以复现相同的视频
result1 = t2v('一只猫在跑步', seed=12345)
result2 = t2v('一只猫在跑步', seed=12345)
# result1 和 result2 应该相同
```

### 4. 错误处理

```python
import lazyllm

t2v = lazyllm.OnlineMultiModalModule(source='qwen', type='text2video')

try:
    result = t2v('一只可爱的猫咪')
    print(f'视频已生成: {result}')
except Exception as e:
    print(f'视频生成失败: {e}')
```

### 5. 解码返回结果

```python
from lazyllm.components.formatter import decode_query_with_filepaths

result = t2v('一只猫在跑步')
decoded = decode_query_with_filepaths(result)
video_path = decoded['files'][0]
print(f'视频保存路径: {video_path}')
```

## 完整示例

### 文本生成视频应用

```python
import lazyllm
from lazyllm.components.formatter import decode_query_with_filepaths

def generate_video(prompt: str, source: str = 'qwen', **kwargs):
    """生成视频的便捷函数"""
    t2v = lazyllm.OnlineMultiModalModule(
        source=source,
        type='text2video'
    )
    
    result = t2v(prompt, **kwargs)
    decoded = decode_query_with_filepaths(result)
    return decoded['files'][0]

# 使用示例
video_path = generate_video(
    '一只可爱的柴犬在雪地里奔跑，背景是美丽的雪山',
    size='1280*720',
    duration=5
)
print(f'视频已保存到: {video_path}')
```

### 图片转视频应用

```python
import lazyllm
from lazyllm.components.formatter import decode_query_with_filepaths

def image_to_video(image_path: str, motion_prompt: str = None, 
                   source: str = 'qwen', **kwargs):
    """将图片转换为视频的便捷函数"""
    i2v = lazyllm.OnlineMultiModalModule(
        source=source,
        type='image2video'
    )
    
    result = i2v(motion_prompt, lazyllm_files=[image_path], **kwargs)
    decoded = decode_query_with_filepaths(result)
    return decoded['files'][0]

# 使用示例
video_path = image_to_video(
    'landscape.jpg',
    '镜头缓慢推进，云层流动',
    size='1280*720'
)
print(f'视频已保存到: {video_path}')
```

## 常见问题

### Q: 视频生成需要多长时间？

A: 视频生成通常需要 1-5 分钟，具体取决于模型和视频参数。API 调用是异步的，SDK 会自动等待任务完成。

### Q: 支持哪些图片格式？

A: 支持常见的图片格式，包括 PNG、JPG、JPEG、WebP 等。图片可以是本地文件路径或 HTTP/HTTPS URL。

### Q: 如何选择合适的模型？

A: 
- **turbo 模型**：生成速度快，适合快速原型和测试
- **plus 模型**：生成质量高，适合正式生产使用

### Q: 生成的视频保存在哪里？

A: 视频会自动下载并保存到临时目录，返回值中包含完整的文件路径。

## 参考链接

- [阿里云万象视频生成 API 文档](https://help.aliyun.com/zh/model-studio/developer-reference/video-generation-api)
- [火山引擎豆包视频生成 API 文档](https://www.volcengine.com/docs/82379/1298451)

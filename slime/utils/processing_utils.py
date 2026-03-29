import base64
import io
import json
import logging
import math
import os
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin

logger = logging.getLogger(__name__)

# Video preprocessing constants — aligned with sglang qwen_vl.py defaults.
# Configurable via environment variables for consistency with sglang server.
IMAGE_FACTOR = 28
FRAME_FACTOR = 2
FPS = float(os.environ.get("VIDEO_FPS", 2.0))
FPS_MIN_FRAMES = int(os.environ.get("VIDEO_MIN_FRAMES", 4))
FPS_MAX_FRAMES = int(os.environ.get("VIDEO_MAX_FRAMES", 768))
VIDEO_MIN_PIXELS = int(os.environ.get("VIDEO_MIN_PIXELS", 128 * 28 * 28))
VIDEO_MAX_PIXELS = int(os.environ.get("VIDEO_MAX_PIXELS_PER_FRAME", 768 * 28 * 28))
# VIDEO_MAX_PIXELS env var is shared with sglang (controls total pixel budget)
VIDEO_TOTAL_PIXELS = int(float(os.environ.get("VIDEO_MAX_PIXELS", 128000 * 28 * 28 * 0.9)))

# Default image patch size for vision-language models
# Note: Qwen3-VL uses 16, Qwen2.5-VL uses 14
# Reference: https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/README.md
DEFAULT_PATCH_SIZE = 14


def load_tokenizer(name_or_path: str, **kwargs):
    return AutoTokenizer.from_pretrained(name_or_path, **kwargs)


def build_processor_kwargs(multimodal_inputs: dict | None = None) -> dict:

    modality_forced = {"return_tensors": "pt"}

    result = dict(multimodal_inputs) if multimodal_inputs else {}

    # return_tensors=None for text (input_ids as lists), "pt" for modality-specific outputs
    result["text_kwargs"] = {
        **result.get("text_kwargs", {}),
        "return_tensors": None,
        "return_mm_token_type_ids": False,
    }
    for key in ("audio_kwargs", "images_kwargs", "videos_kwargs"):
        if key in result:
            result[key] = {**result[key], **modality_forced}
        else:
            result[key] = modality_forced.copy()

    return result


def _try_load_glm4v_processor(name_or_path: str, **kwargs):
    """Fallback: manually construct a Glm4vProcessor for GLM-4.6V / GLM-4.5V models.

    AutoProcessor fails for these models on transformers < 5.0 because
    the Glm46VProcessor / Glm4vMoeProcessor classes are not registered.
    The underlying Glm4vProcessor (non-MoE) works for both variants since
    they share the same vision architecture.
    """
    try:
        from transformers.models.glm4v.image_processing_glm4v import Glm4vImageProcessor
        from transformers.models.glm4v.processing_glm4v import Glm4vProcessor
        from transformers.models.glm4v.video_processing_glm4v import Glm4vVideoProcessor
    except ImportError:
        return None

    pp_path = Path(name_or_path) / "preprocessor_config.json"
    vp_path = Path(name_or_path) / "video_preprocessor_config.json"
    if not pp_path.exists():
        return None

    skip_keys = {"image_processor_type", "processor_class", "video_processor_type"}
    with open(pp_path) as f:
        pp_cfg = {k: v for k, v in json.load(f).items() if k not in skip_keys}
    image_processor = Glm4vImageProcessor(**pp_cfg)

    video_processor = None
    if vp_path.exists():
        with open(vp_path) as f:
            vp_cfg = {k: v for k, v in json.load(f).items() if k not in skip_keys}
        video_processor = Glm4vVideoProcessor(**vp_cfg)

    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    proc = Glm4vProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=tokenizer.chat_template,
    )
    logger.info(f"Loaded Glm4vProcessor manually for {name_or_path}")
    return proc


def load_processor(name_or_path: str, **kwargs):
    try:
        proc = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to load processor from {name_or_path}: {e}")
        proc = None

    # If HF returned a tokenizer instead of a proper processor, discard it.
    if isinstance(proc, PreTrainedTokenizerBase) or not isinstance(proc, ProcessorMixin):
        # Fallback: try to construct a GLM-4.6V / GLM-4.5V processor manually.
        proc = _try_load_glm4v_processor(name_or_path, **kwargs)

    return proc


def _extract_images_from_messages(messages):
    """Extract PIL images from chat messages containing multimodal content.

    Handles base64 strings (with or without data: URI prefix), file paths,
    and PIL Image objects embedded in message content dicts.
    """
    images = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            image_data = item.get("image")
            if image_data is None:
                continue
            if isinstance(image_data, Image.Image):
                images.append(image_data)
            elif isinstance(image_data, str):
                if image_data.startswith("data:"):
                    _, encoded = image_data.split(",", 1)
                    images.append(Image.open(io.BytesIO(base64.b64decode(encoded))))
                else:
                    try:
                        raw = base64.b64decode(image_data)
                        images.append(Image.open(io.BytesIO(raw)))
                    except Exception:
                        # Not base64 — try as file path
                        images.append(Image.open(image_data))
    return images


def process_vision_info(prompt, processor):
    """Extract PIL images (and videos) from the message list for training.

    Tries qwen_vl_utils first (Qwen VL family), falls back to generic
    extraction for other models (e.g. GLM-4.6V).
    """
    try:
        from qwen_vl_utils import process_vision_info as qwen_process_vision_info

        if hasattr(processor.image_processor, "patch_size"):
            image_patch_size = processor.image_processor.patch_size
        else:
            image_patch_size = DEFAULT_PATCH_SIZE
        images, videos = qwen_process_vision_info(prompt, image_patch_size=image_patch_size)
    except Exception:
        # Fallback: generic extraction for non-Qwen models
        images = _extract_images_from_messages(prompt) or None
        videos = None

    return {"images": images, "videos": videos}


def load_image(source) -> Image.Image:
    """Load image from path/URL/base64/PIL without resize (aligned with sglang)."""
    if isinstance(source, Image.Image):
        return source.convert("RGB") if source.mode != "RGB" else source
    if isinstance(source, str):
        if source.startswith("data:"):
            _, encoded = source.split(",", 1)
            return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
        if source.startswith(("http://", "https://")):
            import requests

            return Image.open(io.BytesIO(requests.get(source).content)).convert("RGB")
        if os.path.isfile(source):
            return Image.open(source).convert("RGB")
        try:
            raw = base64.b64decode(source)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Cannot load image from: {source[:100]}...") from e
    raise ValueError(f"Unsupported image source type: {type(source)}")


def prepare_multimodal_for_training(multimodal_inputs: dict) -> dict:
    """Resolve path-based multimodal_inputs into loaded PIL Images / video tensors for training."""
    result = {}
    if multimodal_inputs.get("images"):
        result["images"] = [load_image(img) if isinstance(img, str) else img for img in multimodal_inputs["images"]]
    if multimodal_inputs.get("videos"):
        videos, metadatas = [], []
        for v in multimodal_inputs["videos"]:
            if isinstance(v, str):
                tensor, metadata = preprocess_video(v, IMAGE_FACTOR)
                videos.append(tensor)
                metadatas.append(metadata)
            else:
                videos.append(v)
        result["videos"] = videos
        if metadatas:
            result["videos_kwargs"] = {
                "do_sample_frames": False,
                "video_metadata": metadatas,
            }
    return result


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> tuple[int, int]:
    """Resize dimensions to be divisible by *factor* while staying within pixel budget.

    Aligned with sglang ``smart_resize``.
    """
    if min_pixels is None:
        min_pixels = 4 * factor * factor
    if max_pixels is None:
        max_pixels = 16384 * factor * factor
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def smart_nframes(total_frames: int, video_fps: float) -> int:
    """Calculate number of frames to sample — aligned with sglang ``smart_nframes``."""
    min_frames = ceil_by_factor(FPS_MIN_FRAMES, FRAME_FACTOR)
    max_frames = floor_by_factor(min(FPS_MAX_FRAMES, total_frames), FRAME_FACTOR)
    nframes = total_frames / video_fps * FPS
    if nframes > total_frames:
        logger.warning(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
    nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
    nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


def preprocess_video(
    video_source: str,
    image_factor: int = IMAGE_FACTOR,
) -> tuple:
    """Load and preprocess a video — aligned with sglang ``preprocess_video``.

    Performs frame sampling and BILINEAR resize identical to sglang so that the
    HF processor (called with ``do_sample_frames=False``) produces the same
    ``pixel_values_videos`` / ``video_grid_thw`` as sglang's inference path.

    Returns:
        (video_tensor, video_metadata) where video_tensor is shape (T, C, H, W)
        uint8 and video_metadata is a dict for the HF processor.
    """
    import numpy as np
    import torch
    import torchvision.transforms.functional as F
    from torchvision.transforms import InterpolationMode

    try:
        import decord
    except ImportError as e:
        raise ImportError("decord is required for video preprocessing: pip install decord") from e

    vr = decord.VideoReader(video_source)
    total_frames, video_fps = len(vr), vr.get_avg_fps()

    nframes = smart_nframes(total_frames, video_fps)
    idx = np.linspace(0, total_frames - 1, num=nframes, dtype=np.int64)
    idx = np.unique(idx)

    video = torch.from_numpy(vr.get_batch(idx.tolist()).asnumpy())
    video = video.permute(0, 3, 1, 2)  # THWC -> TCHW

    nframes, _, height, width = video.shape
    max_pixels = max(
        min(VIDEO_MAX_PIXELS, VIDEO_TOTAL_PIXELS / nframes * FRAME_FACTOR),
        int(VIDEO_MIN_PIXELS * 1.05),
    )
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=image_factor,
        min_pixels=VIDEO_MIN_PIXELS,
        max_pixels=max_pixels,
    )
    video = F.resize(video, [resized_height, resized_width], interpolation=InterpolationMode.BILINEAR)

    video_metadata = {
        "fps": video_fps,
        "duration": total_frames / video_fps,
        "total_num_frames": total_frames,
        "frames_indices": idx.tolist(),
    }
    return video, video_metadata


def encode_image_for_rollout_engine(source) -> str:
    """Encode image as base64 for sglang image_data. Accepts PIL Image, path, URL, or base64 string."""
    if isinstance(source, str):
        if source.startswith("data:"):
            return source
        if not os.path.isfile(source) and not source.startswith(("http://", "https://")):
            return f"data:image/png;base64,{source}"
    image = load_image(source)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"


def encode_video_for_rollout_engine(video_source: str, colocate: bool = True) -> str:
    """Encode video for sglang video_data. Returns path directly when colocated."""
    if colocate:
        return video_source
    if video_source.startswith(("http://", "https://")):
        import requests

        data = requests.get(video_source).content
    else:
        with open(video_source, "rb") as f:
            data = f.read()
    return base64.b64encode(data).decode("utf-8")


def prepare_multimodal_for_rollout(multimodal_inputs: dict | None, colocate: bool = True) -> dict:
    """Build sglang payload fields (image_data, video_data) from multimodal_inputs."""
    result = {}
    if not multimodal_inputs:
        return result
    if multimodal_inputs.get("images"):
        result["image_data"] = [encode_image_for_rollout_engine(image) for image in multimodal_inputs["images"]]
    if multimodal_inputs.get("videos"):
        result["video_data"] = [
            encode_video_for_rollout_engine(video, colocate=colocate) for video in multimodal_inputs["videos"]
        ]
    return result

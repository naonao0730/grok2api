"""OpenAI-compatible API router (/v1/*)."""

import base64
import binascii
import mimetypes
from typing import Annotated, Any, AsyncGenerator, AsyncIterable, Literal

import orjson
from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse

from app.control.account.state_machine import is_manageable
from app.platform.auth.middleware import verify_api_key
from app.platform.errors import AppError, ValidationError
from app.platform.logging.logger import logger
from app.platform.storage import image_files_dir, video_files_dir
from app.control.model import registry as model_registry
from app.control.model.spec import ModelSpec
from .schemas import (
    ChatCompletionRequest,
    ImageGenerationRequest,
    VideoConfig,
    ImageConfig,
    ResponsesCreateRequest,
)
from .chat import completions as chat_completions

router = APIRouter(prefix="/v1")
_POOL_ID_TO_NAME = {0: "basic", 1: "super", 2: "heavy"}
_TAG_MODELS = "OpenAI - Models"
_TAG_CHAT = "OpenAI - Chat"
_TAG_RESPONSES = "OpenAI - Responses"
_TAG_IMAGES = "OpenAI - Images"
_TAG_VIDEOS = "OpenAI - Videos"
_TAG_FILES = "OpenAI - Files"


async def _available_pools(request: Request) -> frozenset[str]:
    repo = getattr(request.app.state, "repository", None)
    if repo is None:
        return frozenset()

    snapshot = await repo.runtime_snapshot()
    pools = {
        record.pool
        for record in snapshot.items
        if is_manageable(record)
    }
    return frozenset(pools)


def _model_available_for_pools(spec: ModelSpec, pools: frozenset[str]) -> bool:
    if not spec.enabled:
        return False
    candidates = {
        _POOL_ID_TO_NAME[pool_id]
        for pool_id in spec.pool_candidates()
    }
    return bool(candidates & pools)


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------

@router.get("/models", tags=[_TAG_MODELS], dependencies=[Depends(verify_api_key)])
async def list_models(request: Request):
    import time
    pools = await _available_pools(request)
    models = [
        {
            "id":       m.model_name,
            "object":   "model",
            "created":  int(time.time()),
            "owned_by": "xai",
            "name":     m.public_name,
        }
        for m in model_registry.list_enabled()
        if _model_available_for_pools(m, pools)
    ]
    return JSONResponse({"object": "list", "data": models})


@router.get("/models/{model_id}", tags=[_TAG_MODELS], dependencies=[Depends(verify_api_key)])
async def get_model_endpoint(model_id: str, request: Request):
    import time
    spec = model_registry.get(model_id)
    pools = await _available_pools(request)
    if spec is None or not _model_available_for_pools(spec, pools):
        return JSONResponse(
            {"error": {"message": f"Model {model_id!r} not found", "type": "invalid_request_error"}},
            status_code=404,
        )
    return JSONResponse({
        "id":       spec.model_name,
        "object":   "model",
        "created":  int(time.time()),
        "owned_by": "xai",
        "name":     spec.public_name,
    })


# ---------------------------------------------------------------------------
# SSE streaming helpers
# ---------------------------------------------------------------------------

async def _safe_sse(stream: AsyncIterable[str]) -> AsyncGenerator[str, None]:
    """Wrap an SSE stream, converting exceptions to in-band error events."""
    try:
        async for chunk in stream:
            yield chunk
    except AppError as exc:
        payload = orjson.dumps({"error": exc.to_dict()["error"]}).decode()
        yield f"event: error\ndata: {payload}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:
        payload = orjson.dumps({"error": {"message": str(exc), "type": "server_error"}}).decode()
        yield f"event: error\ndata: {payload}\n\n"
        yield "data: [DONE]\n\n"


_SSE_HEADERS = {"Cache-Control": "no-cache", "Connection": "keep-alive"}


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------

_VALID_ROLES      = {"developer", "system", "user", "assistant", "tool"}
_USER_BLOCK_TYPES = {"text", "image_url", "input_audio", "file"}
_ALLOWED_SIZES    = {"1280x720", "720x1280", "1792x1024", "1024x1792", "1024x1024"}
_EFFORT_VALUES    = {"none", "minimal", "low", "medium", "high", "xhigh"}
_LITE_IMAGE_MODELS = {"grok-imagine-image-lite"}


def _validate_chat(req: ChatCompletionRequest) -> None:
    from app.platform.errors import ValidationError
    spec = model_registry.get(req.model)
    if spec is None or not spec.enabled:
        raise ValidationError(
            f"Model {req.model!r} does not exist or you do not have access to it.",
            param="model", code="model_not_found",
        )
    if not req.messages:
        raise ValidationError("messages cannot be empty", param="messages")
    for i, msg in enumerate(req.messages):
        if msg.role not in _VALID_ROLES:
            raise ValidationError(
                f"role must be one of {sorted(_VALID_ROLES)}",
                param=f"messages.{i}.role",
            )
    if req.temperature is not None and not (0 <= req.temperature <= 2):
        raise ValidationError("temperature must be between 0 and 2", param="temperature")
    if req.top_p is not None and not (0 <= req.top_p <= 1):
        raise ValidationError("top_p must be between 0 and 1", param="top_p")
    if req.reasoning_effort is not None and req.reasoning_effort not in _EFFORT_VALUES:
        raise ValidationError(
            f"reasoning_effort must be one of {sorted(_EFFORT_VALUES)}",
            param="reasoning_effort",
        )


def _validate_image_n(model_name: str, n: int, *, param: str) -> None:
    max_n = 4 if model_name in _LITE_IMAGE_MODELS else 10
    if not (1 <= n <= max_n):
        raise ValidationError(
            f"n must be between 1 and {max_n} for model {model_name!r}",
            param=param,
        )


def _validate_image_edit_n(n: int, *, param: str) -> None:
    if not (1 <= n <= 2):
        raise ValidationError("n must be between 1 and 2 for image edit", param=param)


async def _upload_to_data_uri(upload: UploadFile, *, param: str) -> str:
    raw = await upload.read()
    if not raw:
        raise ValidationError("Uploaded image cannot be empty", param=param)

    mime = (
        (upload.content_type or "").strip().lower()
        or mimetypes.guess_type(upload.filename or "")[0]
        or "application/octet-stream"
    )
    if not mime.startswith("image/"):
        raise ValidationError("Uploaded file must be an image", param=param)

    try:
        blob_b64 = base64.b64encode(raw).decode("ascii")
    except (ValueError, TypeError, binascii.Error) as exc:
        raise ValidationError("Failed to encode uploaded image", param=param) from exc
    return f"data:{mime};base64,{blob_b64}"


def _parse_video_reference_item(value: Any, *, param: str) -> dict[str, str]:
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            raise ValidationError("Reference image cannot be empty", param=param)
        return {"image_url": candidate}

    if isinstance(value, dict):
        block_type = value.get("type")
        if block_type == "image_url":
            image_url = value.get("image_url")
            if isinstance(image_url, dict):
                candidate = str(image_url.get("url") or "").strip()
            else:
                candidate = str(image_url or "").strip()
        else:
            candidate = str(value.get("image_url") or "").strip()
        if not candidate:
            raise ValidationError("Reference image URL cannot be empty", param=param)
        return {"image_url": candidate}

    raise ValidationError("image_reference must be an array of URLs or image_url blocks", param=param)


def _parse_video_references(value: Any, *, param: str) -> list[dict[str, str]]:
    if value in (None, "", []):
        return []

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                value = orjson.loads(stripped)
            except orjson.JSONDecodeError as exc:
                raise ValidationError("image_reference must be a JSON array", param=param) from exc
        else:
            value = [stripped]

    if not isinstance(value, list):
        raise ValidationError("image_reference must be an array", param=param)

    refs: list[dict[str, str]] = []
    for idx, item in enumerate(value):
        refs.append(_parse_video_reference_item(item, param=f"{param}.{idx}"))
    return refs


@router.post("/chat/completions", tags=[_TAG_CHAT], dependencies=[Depends(verify_api_key)])
async def chat_completions_endpoint(req: ChatCompletionRequest):
    _validate_chat(req)

    spec     = model_registry.get(req.model)
    if spec is None:
        raise ValidationError(
            f"Model {req.model!r} does not exist or you do not have access to it.",
            param="model", code="model_not_found",
        )
    messages = [m.model_dump(exclude_none=True) for m in req.messages]

    try:
        # Dispatch by model capability.
        if spec.is_image_edit():
            from .images import edit as img_edit
            cfg    = req.image_config or ImageConfig()
            _validate_image_edit_n(cfg.n or 1, param="image_config.n")
            result = await img_edit(
                model           = req.model,
                messages        = messages,
                n               = cfg.n or 1,
                size            = cfg.size or "1024x1024",
                response_format = cfg.response_format or "url",
                stream          = bool(req.stream),
                chat_format     = True,
            )

        elif spec.is_image():
            from .images import generate as img_gen
            cfg   = req.image_config or ImageConfig()
            size  = cfg.size or "1024x1024"
            fmt   = cfg.response_format or "url"
            n     = cfg.n or 1
            _validate_image_n(req.model, n, param="image_config.n")
            # Extract prompt from last user message.
            prompt = next(
                (m.content for m in reversed(req.messages)
                 if m.role == "user" and isinstance(m.content, str) and m.content.strip()),
                "",
            )
            result = await img_gen(
                model           = req.model,
                prompt          = prompt or "",
                n               = n,
                size            = size,
                response_format = fmt,
                stream          = bool(req.stream),
                chat_format     = True,
            )

        elif spec.is_video():
            from .video import completions as vid_comp
            from .video import normalize_video_size_input as _normalize_video_size_input
            vcfg = req.video_config or VideoConfig()
            from .video import validate_video_length as _validate_video_length
            resolved_seconds = vcfg.video_length or vcfg.seconds or 6
            _validate_video_length(resolved_seconds)
            result = await vid_comp(
                model           = req.model,
                messages        = messages,
                stream          = req.stream,
                seconds         = resolved_seconds,
                size            = _normalize_video_size_input(vcfg.size, vcfg.aspect_ratio),
                resolution_name = vcfg.resolution_name,
                preset          = vcfg.preset,
            )

        else:
            result = await chat_completions(
                model       = req.model,
                messages    = messages,
                stream      = req.stream,
                thinking    = req.thinking,
                tools       = req.tools,
                tool_choice = req.tool_choice,
                temperature = req.temperature or 0.8,
                top_p       = req.top_p or 0.95,
            )

    except AppError:
        raise
    except Exception as exc:
        logger.exception(
            "chat completions endpoint failed: model={} stream={} error={}",
            req.model,
            req.stream,
            exc,
        )
        if req.stream is not False:
            _err_msg = str(exc)  # capture before Python clears the except-scope variable
            async def _err_stream():
                payload = orjson.dumps({"error": {"message": _err_msg, "type": "server_error"}}).decode()
                yield f"event: error\ndata: {payload}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(_err_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)
        raise

    if isinstance(result, dict):
        return JSONResponse(result)
    return StreamingResponse(_safe_sse(result), media_type="text/event-stream", headers=_SSE_HEADERS)


# ---------------------------------------------------------------------------
# /v1/responses  (OpenAI Responses API)
# ---------------------------------------------------------------------------

async def _safe_sse_responses(stream) -> AsyncGenerator[str, None]:
    """SSE wrapper that converts errors to Responses API error events."""
    try:
        async for chunk in stream:
            yield chunk
    except Exception as exc:
        from app.platform.errors import AppError
        if isinstance(exc, AppError):
            err = exc.to_dict()["error"]
        else:
            err = {"message": str(exc), "type": "server_error", "code": None, "param": None}
        payload = orjson.dumps({"type": "error", **err}).decode()
        yield f"event: error\ndata: {payload}\n\n"
        yield "data: [DONE]\n\n"


@router.post("/responses", tags=[_TAG_RESPONSES], dependencies=[Depends(verify_api_key)])
async def responses_endpoint(req: ResponsesCreateRequest):
    from app.platform.config.snapshot import get_config
    from app.platform.errors import ValidationError as _ValidationError

    spec = model_registry.get(req.model)
    if spec is None or not spec.enabled:
        raise _ValidationError(
            f"Model {req.model!r} does not exist or you do not have access to it.",
            param="model", code="model_not_found",
        )
    if not req.input:
        raise _ValidationError("input cannot be empty", param="input")

    cfg        = get_config()
    is_stream  = req.stream if req.stream is not None else cfg.get_bool("features.stream", True)

    # Map reasoning param → emit_think flag.
    # reasoning=None → use config; reasoning.effort="none" → off; otherwise on.
    if req.reasoning is None:
        emit_think = cfg.get_bool("features.thinking", True)
    elif isinstance(req.reasoning, dict) and req.reasoning.get("effort") == "none":
        emit_think = False
    else:
        emit_think = True

    from .responses import create as responses_create
    result = await responses_create(
        model        = req.model,
        input_val    = req.input,
        instructions = req.instructions,
        stream       = is_stream,
        emit_think   = emit_think,
        temperature  = req.temperature or 0.8,
        top_p        = req.top_p or 0.95,
        tools        = req.tools or None,
        tool_choice  = req.tool_choice,
    )

    if isinstance(result, dict):
        return JSONResponse(result)
    return StreamingResponse(
        _safe_sse_responses(result),
        media_type = "text/event-stream",
        headers    = _SSE_HEADERS,
    )


# ---------------------------------------------------------------------------
# /v1/images/generations (standalone image endpoint)
# ---------------------------------------------------------------------------

@router.post("/images/generations", tags=[_TAG_IMAGES], dependencies=[Depends(verify_api_key)])
async def image_generations(req: ImageGenerationRequest):
    spec = model_registry.get(req.model)
    if spec is None or not spec.enabled or not spec.is_image():
        raise ValidationError(f"Model {req.model!r} is not an image model", param="model")
    _validate_image_n(req.model, req.n or 1, param="n")

    from .images import generate as img_gen
    result = await img_gen(
        model           = req.model,
        prompt          = req.prompt,
        n               = req.n or 1,
        size            = req.size or "1024x1024",
        response_format = req.response_format or "url",
        stream          = False,
        chat_format     = False,
    )
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# /v1/videos (OpenAI videos.create surface)
# ---------------------------------------------------------------------------

@router.post("/videos", tags=[_TAG_VIDEOS], dependencies=[Depends(verify_api_key)])
async def videos_create(request: Request):
    from .video import create_video
    from .video import normalize_video_size_input as _normalize_video_size_input
    from .video import resolve_quality_to_resolution_name as _resolve_quality_to_resolution_name

    content_type = (request.headers.get("content-type") or "").lower()

    model = "grok-imagine-video"
    prompt = ""
    seconds: int | str | None = 6
    size: str | None = None
    aspect_ratio: str | None = None
    resolution_name: str | None = None
    quality: str | None = None
    preset: str | None = None
    input_references: list[dict[str, str]] = []

    if "multipart/form-data" in content_type:
        form = await request.form()
        model = str(form.get("model") or model)
        prompt = str(form.get("prompt") or "")
        seconds = form.get("seconds") or 6
        size = str(form.get("size") or "").strip() or None
        aspect_ratio = str(form.get("aspect_ratio") or "").strip() or None
        resolution_name = str(form.get("resolution_name") or "").strip() or None
        quality = str(form.get("quality") or "").strip() or None
        preset = str(form.get("preset") or "").strip() or None
        input_references.extend(_parse_video_references(form.get("image_reference"), param="image_reference"))
        upload = form.get("input_reference")
        if isinstance(upload, UploadFile):
            input_references.append(
                {"image_url": await _upload_to_data_uri(upload, param="input_reference")}
            )
        elif upload not in (None, ""):
            raise ValidationError("input_reference must be an uploaded image", param="input_reference")
    else:
        try:
            payload = await request.json()
        except Exception as exc:
            raise ValidationError("Request body must be JSON or multipart/form-data", param="body") from exc
        if not isinstance(payload, dict):
            raise ValidationError("Request body must be a JSON object", param="body")

        model = str(payload.get("model") or model)
        prompt = str(payload.get("prompt") or "")
        seconds = payload.get("seconds", 6)
        size = str(payload.get("size") or "").strip() or None
        aspect_ratio = str(payload.get("aspect_ratio") or "").strip() or None
        resolution_name = str(payload.get("resolution_name") or "").strip() or None
        quality = str(payload.get("quality") or "").strip() or None
        preset = str(payload.get("preset") or "").strip() or None
        input_references.extend(_parse_video_references(payload.get("image_reference"), param="image_reference"))

        raw_input_reference = payload.get("input_reference")
        if raw_input_reference not in (None, ""):
            if not isinstance(raw_input_reference, dict):
                raise ValidationError("input_reference must be an object with image_url", param="input_reference")
            image_url = str(raw_input_reference.get("image_url") or "").strip()
            if not image_url:
                raise ValidationError("input_reference.image_url is required", param="input_reference.image_url")
            input_references.append({"image_url": image_url})

    if quality and not resolution_name:
        resolution_name = _resolve_quality_to_resolution_name(quality)

    result = await create_video(
        model=model,
        prompt=prompt,
        seconds=seconds,
        size=_normalize_video_size_input(size, aspect_ratio),
        resolution_name=resolution_name,
        quality=quality,
        preset=preset,
        input_reference=input_references or None,
    )
    return JSONResponse(result)


@router.get("/videos/{video_id}", tags=[_TAG_VIDEOS], dependencies=[Depends(verify_api_key)])
async def videos_retrieve(video_id: str):
    from .video import retrieve
    return JSONResponse(await retrieve(video_id))


@router.get("/videos/{video_id}/content", tags=[_TAG_VIDEOS], dependencies=[Depends(verify_api_key)])
async def videos_content(video_id: str):
    from .video import content_path
    path = await content_path(video_id)
    return FileResponse(path, media_type="video/mp4", filename=f"{video_id}.mp4")


# ---------------------------------------------------------------------------
# /v1/images/edits (standalone image-edit endpoint)
# ---------------------------------------------------------------------------

@router.post("/images/edits", tags=[_TAG_IMAGES], dependencies=[Depends(verify_api_key)])
async def image_edits(
    model: Annotated[str, Form(...)],
    prompt: Annotated[str, Form(...)],
    image: Annotated[list[UploadFile], File(..., alias="image[]")],
    mask: Annotated[UploadFile | None, File()] = None,
    n: Annotated[int, Form()] = 1,
    size: Annotated[str, Form()] = "1024x1024",
    response_format: Annotated[str, Form()] = "url",
):
    spec = model_registry.get(model)
    if spec is None or not spec.enabled or not spec.is_image_edit():
        raise ValidationError(f"Model {model!r} is not an image-edit model", param="model")
    if mask is not None:
        raise ValidationError("mask is not supported yet", param="mask")
    _validate_image_edit_n(n, param="n")

    from .images import edit as img_edit
    image_inputs = [
        await _upload_to_data_uri(item, param=f"image.{index}")
        for index, item in enumerate(image)
    ]
    # Wrap input into a single-message conversation.
    content = [{"type": "text", "text": prompt}]
    content.extend(
        {"type": "image_url", "image_url": {"url": image_input}}
        for image_input in image_inputs
    )
    messages = [{"role": "user", "content": content}]
    result = await img_edit(
        model           = model,
        messages        = messages,
        n               = n,
        size            = size,
        response_format = response_format,
        stream          = False,
        chat_format     = False,
    )
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# /v1/files/image — serve locally saved images
# ---------------------------------------------------------------------------

@router.get("/files/video", tags=[_TAG_FILES])
async def serve_video(id: str = Query(..., description="Video file ID")):
    """Serve a locally cached video by file ID."""
    import re

    if not re.fullmatch(r"[0-9a-f\-]{16,36}", id):
        raise ValidationError("Invalid file ID", param="id")

    path = video_files_dir() / f"{id}.mp4"
    if path.exists():
        return FileResponse(path, media_type="video/mp4")

    raise ValidationError(f"Video {id!r} not found", param="id")


@router.get("/files/image", tags=[_TAG_FILES])
async def serve_image(id: str = Query(..., description="Image file ID")):
    """Serve a locally cached image by file ID."""
    import re

    if not re.fullmatch(r"[0-9a-f\-]{16,36}", id):
        raise ValidationError("Invalid file ID", param="id")

    img_dir = image_files_dir()
    for ext in (".jpg", ".png"):
        path = img_dir / f"{id}{ext}"
        if path.exists():
            mime = "image/png" if ext == ".png" else "image/jpeg"
            return FileResponse(path, media_type=mime)

    raise ValidationError(f"Image {id!r} not found", param="id")


__all__ = ["router"]

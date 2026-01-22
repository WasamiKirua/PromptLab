from __future__ import annotations

import json
import mimetypes
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, File, Form, Request, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from google import genai
from google.genai import types

from captioning_utils import generate_caption_and_prompt_from_image, generate_prompt_from_text

load_dotenv()

BASE_DIR = Path(__file__).parent
ENV_PATH = BASE_DIR / ".env"
TEMPLATES_DIR = BASE_DIR / "templates"
IMAGES_GENERATED = BASE_DIR / "static" / "images_generated"
IMAGES_GENERATED.mkdir(exist_ok=True)

def load_env_file() -> tuple[dict[str, str], bool]:
    if not ENV_PATH.exists():
        return {}, False
    values: dict[str, str] = {}
    for line in ENV_PATH.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip()
    return values, True


def update_env_file(updates: dict[str, str]) -> None:
    existing_lines = []
    if ENV_PATH.exists():
        existing_lines = ENV_PATH.read_text().splitlines()

    remaining = {key: value for key, value in updates.items()}
    output_lines = []
    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            output_lines.append(line)
            continue
        key, _ = stripped.split("=", 1)
        key = key.strip()
        if key in remaining:
            value = remaining.pop(key)
            if value:
                output_lines.append(f"{key}={value}")
            continue
        output_lines.append(line)

    for key, value in remaining.items():
        if value:
            output_lines.append(f"{key}={value}")

    if output_lines:
        ENV_PATH.write_text("\n".join(output_lines).strip() + "\n")


def _clean_env_value(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        return cleaned[1:-1]
    return cleaned


def _resolve_service_key(service: str, env_values: dict[str, str]) -> str:
    service = service.lower()
    if service == "grok":
        return _clean_env_value(env_values.get("GROK_API_KEY", ""))
    if service == "openai":
        return _clean_env_value(env_values.get("OPENAI_API_KEY", ""))
    gemini_key = env_values.get("GEMINI_API_KEY") or env_values.get("GOOGLE_AI_STUDIO_API") or ""
    return _clean_env_value(gemini_key)


def _set_nested_value(data: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cursor: Dict[str, Any] = data
    for key in keys[:-1]:
        next_value = cursor.get(key)
        if not isinstance(next_value, dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def _parse_list(value: str) -> list[str]:
    parts = [item.strip() for item in value.replace("\n", ",").split(",")]
    return [item for item in parts if item]


def _with_cache_buster(url: str) -> str:
    return f"{url}?v={int(time.time())}"


def _list_generated_images() -> list[dict[str, str]]:
    images: list[dict[str, str]] = []
    if not IMAGES_GENERATED.exists():
        return images

    for path in IMAGES_GENERATED.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            continue
        version = int(path.stat().st_mtime)
        images.append(
            {
                "name": path.name,
                "url": f"/static/images_generated/{path.name}?v={version}",
                "mtime": str(version),
            }
        )
    images.sort(key=lambda item: int(item["mtime"]), reverse=True)
    for item in images:
        item.pop("mtime", None)
    return images


def generate_variation_from_template_json(
    template_json: str | Path | Dict[str, Any],
    *,
    out_image_path: str | Path = "variation.png",
    model: str = "gemini-3-pro-image-preview",
    aspect_ratio: Optional[str] = None,
    image_size: Optional[str] = None,
) -> Path:
    out_image_path = Path(out_image_path).expanduser().resolve()

    if isinstance(template_json, (str, Path)):
        template_path = Path(template_json).expanduser().resolve()
        spec = json.loads(template_path.read_text(encoding="utf-8"))
    else:
        spec = template_json

    prompt = (
        "Generate a high-quality image using this JSON specification as the full guide. "
        "Do not infer anything outside the provided JSON.\n"
        f"{json.dumps(spec, ensure_ascii=False)}"
    )

    image_config_kwargs: Dict[str, Any] = {}
    if aspect_ratio:
        image_config_kwargs["aspect_ratio"] = aspect_ratio
    if image_size:
        image_config_kwargs["image_size"] = image_size

    safety_settings = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
    ]

    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    )
    response = client.models.generate_content(
        model=model,
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            image_config=types.ImageConfig(**image_config_kwargs) if image_config_kwargs else None,
            safety_settings=safety_settings,
            tools=[types.Tool(googleSearch=types.GoogleSearch())],
        ),
    )

    for part in getattr(response, "parts", None) or []:
        if part.inline_data and part.inline_data.data:
            file_extension = mimetypes.guess_extension(part.inline_data.mime_type) or ".png"
            final_path = out_image_path.with_suffix(file_extension)
            final_path.write_bytes(part.inline_data.data)
            return final_path

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            if part.inline_data and part.inline_data.data:
                file_extension = mimetypes.guess_extension(part.inline_data.mime_type) or ".png"
                final_path = out_image_path.with_suffix(file_extension)
                final_path.write_bytes(part.inline_data.data)
                return final_path

    raise RuntimeError("No image returned by the model.")


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
        },
    )


@app.post("/settings")
async def update_settings(request: Request) -> JSONResponse:
    form = await request.form()
    service = (form.get("service") or "gemini").strip().lower()
    api_key = (form.get("api_key") or "").strip()

    updates = {}
    if service == "grok":
        updates["GROK_API_KEY"] = api_key
    elif service == "openai":
        updates["OPENAI_API_KEY"] = api_key
    else:
        updates["GEMINI_API_KEY"] = api_key

    update_env_file(updates)
    for key, value in updates.items():
        if value:
            os.environ[key] = value
    return JSONResponse({"status": "ok"})


@app.get("/nano-banana", response_class=HTMLResponse)
async def nano_banana(request: Request) -> HTMLResponse:
    recent_images = _list_generated_images()[:6]
    return templates.TemplateResponse(
        "nano_banana.html",
        {
            "request": request,
            "recent_images": recent_images,
        },
    )


@app.get("/caption-to-prompt", response_class=HTMLResponse)
async def caption_to_prompt(request: Request) -> HTMLResponse:
    env_values, _ = load_env_file()
    selected_service = "gemini"
    service_api_key = _resolve_service_key(selected_service, env_values)
    gemini_api_key = _clean_env_value(
        env_values.get("GEMINI_API_KEY") or env_values.get("GOOGLE_AI_STUDIO_API") or ""
    )
    grok_api_key = _clean_env_value(env_values.get("GROK_API_KEY", ""))
    openai_api_key = _clean_env_value(env_values.get("OPENAI_API_KEY", ""))
    return templates.TemplateResponse(
        "caption_to_prompt.html",
        {
            "request": request,
            "selected_service": selected_service,
            "service_api_key": service_api_key,
            "gemini_api_key": gemini_api_key,
            "grok_api_key": grok_api_key,
            "openai_api_key": openai_api_key,
        },
    )


@app.get("/text-to-prompt", response_class=HTMLResponse)
async def text_to_prompt(request: Request) -> HTMLResponse:
    env_values, _ = load_env_file()
    selected_service = "gemini"
    service_api_key = _resolve_service_key(selected_service, env_values)
    gemini_api_key = _clean_env_value(
        env_values.get("GEMINI_API_KEY") or env_values.get("GOOGLE_AI_STUDIO_API") or ""
    )
    grok_api_key = _clean_env_value(env_values.get("GROK_API_KEY", ""))
    openai_api_key = _clean_env_value(env_values.get("OPENAI_API_KEY", ""))
    return templates.TemplateResponse(
        "text_to_prompt.html",
        {
            "request": request,
            "selected_service": selected_service,
            "service_api_key": service_api_key,
            "gemini_api_key": gemini_api_key,
            "grok_api_key": grok_api_key,
            "openai_api_key": openai_api_key,
        },
    )


@app.post("/caption-to-prompt")
async def caption_to_prompt_generate(
    image: UploadFile = File(...),
    target_base_model: str = Form("z-image"),
    service: str = Form(""),
    allow_nsfw: str = Form("false"),
) -> JSONResponse:
    if not image or not image.filename:
        raise HTTPException(status_code=400, detail="Image is required.")
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported.")

    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    env_values, _ = load_env_file()
    selected_service = (service or "").strip().lower() or "gemini"
    api_key = _resolve_service_key(selected_service, env_values)
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing API key for selected service.")

    suffix = Path(image.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(payload)
        tmp_path = tmp_file.name

    try:
        prompt = generate_caption_and_prompt_from_image(
            tmp_path,
            target_base_model,
            selected_service,
            api_key,
            allow_nsfw=allow_nsfw.lower() == "true",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass

    return JSONResponse({"prompt": prompt})


@app.get("/sidneysteps", response_class=HTMLResponse)
async def sidney_steps(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "sidneysteps.html",
        {
            "request": request,
        },
    )


@app.post("/sidneysteps", response_class=HTMLResponse)
async def sidney_steps_generate(request: Request) -> HTMLResponse:
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    try:
        base_path = Path("json_templates_in/sidney_steps.json")
        sidney_steps_data = json.loads(base_path.read_text(encoding="utf-8"))
        output_image_path = IMAGES_GENERATED / "sidneysteps.png"

        result_path = generate_variation_from_template_json(
            template_json=sidney_steps_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "sidneysteps.html",
        {
            "request": request,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.get("/japansecretary", response_class=HTMLResponse)
async def japan_secretary(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "japansecretary.html",
        {
            "request": request,
        },
    )


@app.post("/japansecretary", response_class=HTMLResponse)
async def japan_secretary_generate(request: Request) -> HTMLResponse:
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    try:
        base_path = Path("json_templates_in/japan_secretary.json")
        japan_secretary_data = json.loads(base_path.read_text(encoding="utf-8"))
        output_image_path = IMAGES_GENERATED / "japansecretary.png"

        result_path = generate_variation_from_template_json(
            template_json=japan_secretary_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "japansecretary.html",
        {
            "request": request,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.get("/coolplants", response_class=HTMLResponse)
async def cool_plants(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "coolplants.html",
        {
            "request": request,
        },
    )


@app.post("/coolplants", response_class=HTMLResponse)
async def cool_plants_generate(request: Request) -> HTMLResponse:
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    try:
        base_path = Path("json_templates_in/cool_plants.json")
        cool_plants_data = json.loads(base_path.read_text(encoding="utf-8"))
        output_image_path = IMAGES_GENERATED / "coolplants.png"

        result_path = generate_variation_from_template_json(
            template_json=cool_plants_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "coolplants.html",
        {
            "request": request,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.get("/onterrace", response_class=HTMLResponse)
async def on_terrace(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "onterrace.html",
        {
            "request": request,
        },
    )


@app.post("/onterrace", response_class=HTMLResponse)
async def on_terrace_generate(request: Request) -> HTMLResponse:
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    try:
        base_path = Path("json_templates_in/on_terrace.json")
        on_terrace_data = json.loads(base_path.read_text(encoding="utf-8"))
        output_image_path = IMAGES_GENERATED / "onterrace.png"

        result_path = generate_variation_from_template_json(
            template_json=on_terrace_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "onterrace.html",
        {
            "request": request,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.get("/closeup", response_class=HTMLResponse)
async def closeup(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "closeup.html",
        {
            "request": request,
        },
    )


@app.post("/closeup", response_class=HTMLResponse)
async def closeup_generate(request: Request) -> HTMLResponse:
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    try:
        base_path = Path("json_templates_in/closeup.json")
        closeup_data = json.loads(base_path.read_text(encoding="utf-8"))
        output_image_path = IMAGES_GENERATED / "closeup.png"

        result_path = generate_variation_from_template_json(
            template_json=closeup_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "closeup.html",
        {
            "request": request,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.get("/womancomics", response_class=HTMLResponse)
async def woman_comics(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "womancomics.html",
        {
            "request": request,
        },
    )


@app.post("/womancomics", response_class=HTMLResponse)
async def woman_comics_generate(request: Request) -> HTMLResponse:
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    try:
        base_path = Path("json_templates_in/woman_comics.json")
        woman_comics_data = json.loads(base_path.read_text(encoding="utf-8"))
        output_image_path = IMAGES_GENERATED / "womancomics.png"

        result_path = generate_variation_from_template_json(
            template_json=woman_comics_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "womancomics.html",
        {
            "request": request,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.get("/sadiesink", response_class=HTMLResponse)
async def sadie_sink(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "sadiesink.html",
        {
            "request": request,
        },
    )


@app.post("/sadiesink", response_class=HTMLResponse)
async def sadie_sink_generate(request: Request) -> HTMLResponse:
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    try:
        base_path = Path("json_templates_in/sadie_sink.json")
        sadie_sink_data = json.loads(base_path.read_text(encoding="utf-8"))
        output_image_path = IMAGES_GENERATED / "sadiesink.png"

        result_path = generate_variation_from_template_json(
            template_json=sadie_sink_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "sadiesink.html",
        {
            "request": request,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.get("/breakinginternet", response_class=HTMLResponse)
async def breaking_internet(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "breakinginternet.html",
        {
            "request": request,
        },
    )


@app.post("/breakinginternet", response_class=HTMLResponse)
async def breaking_internet_generate(request: Request) -> HTMLResponse:
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    try:
        base_path = Path("json_templates_in/breaking_internet.json")
        breaking_internet_data = json.loads(base_path.read_text(encoding="utf-8"))
        output_image_path = IMAGES_GENERATED / "breakinginternet.png"

        result_path = generate_variation_from_template_json(
            template_json=breaking_internet_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "breakinginternet.html",
        {
            "request": request,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.get("/greekpaint", response_class=HTMLResponse)
async def greek_paint(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "greekpaint.html",
        {
            "request": request,
        },
    )


@app.post("/greekpaint", response_class=HTMLResponse)
async def greek_paint_generate(request: Request) -> HTMLResponse:
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    try:
        base_path = Path("json_templates_in/greek_paint.json")
        greek_paint_data = json.loads(base_path.read_text(encoding="utf-8"))
        output_image_path = IMAGES_GENERATED / "greekpaint.png"

        result_path = generate_variation_from_template_json(
            template_json=greek_paint_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "greekpaint.html",
        {
            "request": request,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.get("/twins", response_class=HTMLResponse)
async def twins(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "twins.html",
        {
            "request": request,
        },
    )


@app.post("/twins", response_class=HTMLResponse)
async def twins_generate(request: Request) -> HTMLResponse:
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    try:
        base_path = Path("json_templates_in/twins.json")
        twins_data = json.loads(base_path.read_text(encoding="utf-8"))
        output_image_path = IMAGES_GENERATED / "twins.png"

        result_path = generate_variation_from_template_json(
            template_json=twins_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "twins.html",
        {
            "request": request,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.get("/eastasian", response_class=HTMLResponse)
async def east_asian(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "eastasian.html",
        {
            "request": request,
        },
    )


@app.post("/eastasian", response_class=HTMLResponse)
async def east_asian_generate(request: Request) -> HTMLResponse:
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    try:
        base_path = Path("json_templates_in/east_asian.json")
        east_asian_data = json.loads(base_path.read_text(encoding="utf-8"))
        output_image_path = IMAGES_GENERATED / "eastasian.png"

        result_path = generate_variation_from_template_json(
            template_json=east_asian_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "eastasian.html",
        {
            "request": request,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.get("/insidecar", response_class=HTMLResponse)
async def inside_car(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "insidecar.html",
        {
            "request": request,
        },
    )


@app.post("/insidecar", response_class=HTMLResponse)
async def inside_car_generate(request: Request) -> HTMLResponse:
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    try:
        base_path = Path("json_templates_in/inside_car.json")
        inside_car_data = json.loads(base_path.read_text(encoding="utf-8"))
        output_image_path = IMAGES_GENERATED / "insidecar.png"

        result_path = generate_variation_from_template_json(
            template_json=inside_car_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "insidecar.html",
        {
            "request": request,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.get("/justconfidence", response_class=HTMLResponse)
async def just_confidence(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "justconfidence.html",
        {
            "request": request,
        },
    )


@app.post("/justconfidence", response_class=HTMLResponse)
async def just_confidence_generate(request: Request) -> HTMLResponse:
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    try:
        base_path = Path("json_templates_in/just_confidence.json")
        just_confidence_data = json.loads(base_path.read_text(encoding="utf-8"))
        output_image_path = IMAGES_GENERATED / "justconfidence.png"

        result_path = generate_variation_from_template_json(
            template_json=just_confidence_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "justconfidence.html",
        {
            "request": request,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.get("/editorialfashion", response_class=HTMLResponse)
async def editorial_fashion(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "editorialfashion.html",
        {
            "request": request,
        },
    )


@app.post("/editorialfashion", response_class=HTMLResponse)
async def editorial_fashion_generate(request: Request) -> HTMLResponse:
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    try:
        base_path = Path("json_templates_in/editorial_fashion.json")
        editorial_fashion_data = json.loads(base_path.read_text(encoding="utf-8"))
        output_image_path = IMAGES_GENERATED / "editorialfashion.png"

        result_path = generate_variation_from_template_json(
            template_json=editorial_fashion_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "editorialfashion.html",
        {
            "request": request,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.get("/withflowers", response_class=HTMLResponse)
async def with_flowers(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "withflowers.html",
        {
            "request": request,
        },
    )


@app.post("/withflowers", response_class=HTMLResponse)
async def with_flowers_generate(request: Request) -> HTMLResponse:
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    try:
        base_path = Path("json_templates_in/with_flowers.json")
        with_flowers_data = json.loads(base_path.read_text(encoding="utf-8"))
        output_image_path = IMAGES_GENERATED / "withflowers.png"

        result_path = generate_variation_from_template_json(
            template_json=with_flowers_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "withflowers.html",
        {
            "request": request,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.get("/marmorlook", response_class=HTMLResponse)
async def marmor_look(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "marmorlook.html",
        {
            "request": request,
        },
    )


@app.post("/marmorlook", response_class=HTMLResponse)
async def marmor_look_generate(request: Request) -> HTMLResponse:
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    try:
        base_path = Path("json_templates_in/marmor_look.json")
        marmor_look_data = json.loads(base_path.read_text(encoding="utf-8"))
        output_image_path = IMAGES_GENERATED / "marmorlook.png"

        result_path = generate_variation_from_template_json(
            template_json=marmor_look_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "marmorlook.html",
        {
            "request": request,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.get("/scandinavian", response_class=HTMLResponse)
async def scandinavian(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "scandinavian.html",
        {
            "request": request,
        },
    )


@app.post("/scandinavian", response_class=HTMLResponse)
async def scandinavian_generate(request: Request) -> HTMLResponse:
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    try:
        base_path = Path("json_templates_in/scandinavian.json")
        scandinavian_data = json.loads(base_path.read_text(encoding="utf-8"))
        output_image_path = IMAGES_GENERATED / "scandinavian.png"

        result_path = generate_variation_from_template_json(
            template_json=scandinavian_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "scandinavian.html",
        {
            "request": request,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.get("/scandinaviancontroller", response_class=HTMLResponse)
async def scandinavian_controller(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "scandinaviancontroller.html",
        {
            "request": request,
        },
    )


@app.post("/scandinaviancontroller", response_class=HTMLResponse)
async def scandinavian_controller_generate(request: Request) -> HTMLResponse:
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    try:
        base_path = Path("json_templates_in/scandinavian_controller.json")
        scandinavian_controller_data = json.loads(base_path.read_text(encoding="utf-8"))
        output_image_path = IMAGES_GENERATED / "scandinaviancontroller.png"

        result_path = generate_variation_from_template_json(
            template_json=scandinavian_controller_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "scandinaviancontroller.html",
        {
            "request": request,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.get("/playfulselfy", response_class=HTMLResponse)
async def playful_selfy(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "playfulselfy.html",
        {
            "request": request,
        },
    )


@app.post("/playfulselfy", response_class=HTMLResponse)
async def playful_selfy_generate(request: Request) -> HTMLResponse:
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    try:
        base_path = Path("json_templates_in/playful_selfy.json")
        playful_selfy_data = json.loads(base_path.read_text(encoding="utf-8"))
        output_image_path = IMAGES_GENERATED / "playfulselfy.png"

        result_path = generate_variation_from_template_json(
            template_json=playful_selfy_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "playfulselfy.html",
        {
            "request": request,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.get("/attractasian", response_class=HTMLResponse)
async def attract_asian(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "attractasian.html",
        {
            "request": request,
        },
    )


@app.post("/attractasian", response_class=HTMLResponse)
async def attract_asian_generate(request: Request) -> HTMLResponse:
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    try:
        base_path = Path("json_templates_in/attract_asian.json")
        attract_asian_data = json.loads(base_path.read_text(encoding="utf-8"))
        output_image_path = IMAGES_GENERATED / "attractasian.png"

        result_path = generate_variation_from_template_json(
            template_json=attract_asian_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "attractasian.html",
        {
            "request": request,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.get("/gumwoman", response_class=HTMLResponse)
async def gum_woman(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "gumwoman.html",
        {
            "request": request,
        },
    )


@app.post("/gumwoman", response_class=HTMLResponse)
async def gum_woman_generate(request: Request) -> HTMLResponse:
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    try:
        base_path = Path("json_templates_in/gum_woman.json")
        gum_woman_data = json.loads(base_path.read_text(encoding="utf-8"))
        output_image_path = IMAGES_GENERATED / "gumwoman.png"

        result_path = generate_variation_from_template_json(
            template_json=gum_woman_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "gumwoman.html",
        {
            "request": request,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.get("/ustia", response_class=HTMLResponse)
async def ustia(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "ustia.html",
        {
            "request": request,
        },
    )


@app.post("/ustia", response_class=HTMLResponse)
async def ustia_generate(request: Request) -> HTMLResponse:
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    try:
        base_path = Path("json_templates_in/ustia.json")
        ustia_data = json.loads(base_path.read_text(encoding="utf-8"))
        output_image_path = IMAGES_GENERATED / "ustia.png"

        result_path = generate_variation_from_template_json(
            template_json=ustia_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "ustia.html",
        {
            "request": request,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/text-to-prompt")
async def text_to_prompt_generate(
    text: str = Form(""),
    target_base_model: str = Form("z-image"),
    service: str = Form(""),
) -> JSONResponse:
    cleaned_text = (text or "").strip()
    if not cleaned_text:
        raise HTTPException(status_code=400, detail="Text is required.")

    env_values, _ = load_env_file()
    selected_service = (service or "").strip().lower() or "gemini"
    api_key = _resolve_service_key(selected_service, env_values)
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing API key for selected service.")

    try:
        prompt = generate_prompt_from_text(
            cleaned_text,
            target_base_model,
            selected_service,
            api_key,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse({"prompt": prompt})


@app.post("/generatefromjson", response_class=HTMLResponse)
async def generate_from_json(
    request: Request,
    json_file: UploadFile = File(...),
) -> HTMLResponse:
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    try:
        if not json_file or not json_file.filename:
            raise ValueError("No JSON file uploaded.")

        if not json_file.filename.lower().endswith(".json"):
            raise ValueError("Only .json files are supported.")

        raw = await json_file.read()
        if not raw:
            raise ValueError("Uploaded JSON file is empty.")

        try:
            spec = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSON file: {exc}") from exc

        output_name = f"json_upload_{int(time.time())}.png"
        output_image_path = IMAGES_GENERATED / output_name

        result_path = generate_variation_from_template_json(
            template_json=spec,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "nano_banana.html",
        {
            "request": request,
            "recent_images": _list_generated_images()[:6],
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.get("/gallery", response_class=HTMLResponse)
async def gallery(request: Request) -> HTMLResponse:
    images = _list_generated_images()
    return templates.TemplateResponse(
        "gallery.html",
        {
            "request": request,
            "images": images,
        },
    )


@app.post("/gallery/delete")
async def delete_generated_image(filename: str = Form(...)) -> RedirectResponse:
    safe_name = Path(filename).name
    target = IMAGES_GENERATED / safe_name
    if target.exists() and target.is_file():
        target.unlink()
    return RedirectResponse(url="/gallery", status_code=303)


## Nano Banana Templates
@app.get("/liveinbedroom", response_class=HTMLResponse)
async def livein_bedroom(request: Request) -> HTMLResponse:
    livein_bedroom_data = json.load(open("json_templates_in/modern_live_in_bedroom.json"))
    return templates.TemplateResponse(
        "liveinbedroom.html",
        {
            "request": request,
            "livein_bedroom_data": livein_bedroom_data,
        },
    )


@app.post("/liveinbedroom", response_class=HTMLResponse)
async def liveinbedroom_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/modern_live_in_bedroom.json")
    livein_bedroom_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    list_fields = {
        "scene.environment",
        "camera_perspective.imperfections",
        "subject.outfit.details",
        "mood_keywords",
    }

    for key, value in form.items():
        if value is None:
            continue
        if key in list_fields:
            _set_nested_value(livein_bedroom_data, key, _parse_list(str(value)))
        else:
            _set_nested_value(livein_bedroom_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/liveinbedroom.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=livein_bedroom_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(f"/static/images_generated/{Path(result_path).name}")
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "liveinbedroom.html",
        {
            "request": request,
            "livein_bedroom_data": livein_bedroom_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.get("/mirrorselfiedebroom", response_class=HTMLResponse)
async def mirrorselfie_bedroom(request: Request) -> HTMLResponse:
    mirror_selfie_bedroom_data = json.load(open("json_templates_in/mirror_selfie_bedroom.json"))
    return templates.TemplateResponse(
        "mirrorselfiebedroom.html",
        {
            "request": request,
            "mirror_selfie_bedroom_data": mirror_selfie_bedroom_data,
        },
    )


@app.post("/mirrorselfiedebroom", response_class=HTMLResponse)
async def mirrorselfiebedroom_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/mirror_selfie_bedroom.json")
    mirror_selfie_bedroom_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    for key, value in form.items():
        if value is None:
            continue
        _set_nested_value(mirror_selfie_bedroom_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/mirrorselfiebedroom.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=mirror_selfie_bedroom_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(f"/static/images_generated/{Path(result_path).name}")
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "mirrorselfiebedroom.html",
        {
            "request": request,
            "mirror_selfie_bedroom_data": mirror_selfie_bedroom_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.get("/privatepoolnight", response_class=HTMLResponse)
async def privatepoolnight(request: Request) -> HTMLResponse:
    private_pool_night_data = json.load(open("json_templates_in/private_pool_night.json"))
    return templates.TemplateResponse(
        "privatepoolnight.html",
        {
            "request": request,
            "private_pool_night_data": private_pool_night_data,
        },
    )


@app.get("/winterfashion", response_class=HTMLResponse)
async def winterfashion(request: Request) -> HTMLResponse:
    winter_fashion_data = json.load(open("json_templates_in/winter_fashion.json"))
    return templates.TemplateResponse(
        "winterfashion.html",
        {
            "request": request,
            "winter_fashion_data": winter_fashion_data,
        },
    )


@app.get("/analogmirror", response_class=HTMLResponse)
async def analogmirror(request: Request) -> HTMLResponse:
    analog_mirror_data = json.load(open("json_templates_in/analog_mirror.json"))
    return templates.TemplateResponse(
        "analogmirror.html",
        {
            "request": request,
            "analog_mirror_data": analog_mirror_data,
        },
    )


@app.get("/smallbedroom", response_class=HTMLResponse)
async def smallbedroom(request: Request) -> HTMLResponse:
    small_bedroom_data = json.load(open("json_templates_in/small_bedroom.json"))
    return templates.TemplateResponse(
        "smallbedroom.html",
        {
            "request": request,
            "small_bedroom_data": small_bedroom_data,
        },
    )


@app.get("/rusticbedroom", response_class=HTMLResponse)
async def rusticbedroom(request: Request) -> HTMLResponse:
    rustic_bedroom_data = json.load(open("json_templates_in/rustic_bedroom.json"))
    return templates.TemplateResponse(
        "rusticbedroom.html",
        {
            "request": request,
            "rustic_bedroom_data": rustic_bedroom_data,
        },
    )


@app.get("/otakucorner", response_class=HTMLResponse)
async def otakucorner(request: Request) -> HTMLResponse:
    otaku_corner_data = json.load(open("json_templates_in/otaku_corner.json"))
    return templates.TemplateResponse(
        "otakucorner.html",
        {
            "request": request,
            "otaku_corner_data": otaku_corner_data,
        },
    )


@app.get("/nightportrait", response_class=HTMLResponse)
async def nightportrait(request: Request) -> HTMLResponse:
    night_portrait_data = json.load(open("json_templates_in/night_portrait.json"))
    return templates.TemplateResponse(
        "nightportrait.html",
        {
            "request": request,
            "night_portrait_data": night_portrait_data,
        },
    )


@app.get("/edgywoman", response_class=HTMLResponse)
async def edgywoman(request: Request) -> HTMLResponse:
    edgy_woman_data = json.load(open("json_templates_in/edgy_woman.json"))
    return templates.TemplateResponse(
        "edgywoman.html",
        {
            "request": request,
            "edgy_woman_data": edgy_woman_data,
        },
    )


@app.get("/allurgingwoman", response_class=HTMLResponse)
async def allurgingwoman(request: Request) -> HTMLResponse:
    allurging_woman_data = json.load(open("json_templates_in/allurging_woman.json"))
    return templates.TemplateResponse(
        "allurgingwoman.html",
        {
            "request": request,
            "allurging_woman_data": allurging_woman_data,
        },
    )


@app.get("/minidress", response_class=HTMLResponse)
async def minidress(request: Request) -> HTMLResponse:
    mini_dress_data = json.load(open("json_templates_in/mini_dress.json"))
    return templates.TemplateResponse(
        "minidress.html",
        {
            "request": request,
            "mini_dress_data": mini_dress_data,
        },
    )


@app.get("/whitesend", response_class=HTMLResponse)
async def whitesend(request: Request) -> HTMLResponse:
    white_send_data = json.load(open("json_templates_in/white_send.json"))
    return templates.TemplateResponse(
        "whitesend.html",
        {
            "request": request,
            "white_send_data": white_send_data,
        },
    )


@app.get("/whitetop", response_class=HTMLResponse)
async def whitetop(request: Request) -> HTMLResponse:
    white_top_data = json.load(open("json_templates_in/white_top.json"))
    return templates.TemplateResponse(
        "whitetop.html",
        {
            "request": request,
            "white_top_data": white_top_data,
        },
    )


@app.get("/luxuryaesthetic", response_class=HTMLResponse)
async def luxuryaesthetic(request: Request) -> HTMLResponse:
    luxury_aesthetic_data = json.load(open("json_templates_in/luxury_aesthetic.json"))
    return templates.TemplateResponse(
        "luxuryaesthetic.html",
        {
            "request": request,
            "luxury_aesthetic_data": luxury_aesthetic_data,
        },
    )


@app.get("/atleticwoman", response_class=HTMLResponse)
async def atleticwoman(request: Request) -> HTMLResponse:
    atletic_woman_data = json.load(open("json_templates_in/atletic_woman.json"))
    return templates.TemplateResponse(
        "atleticwoman.html",
        {
            "request": request,
            "atletic_woman_data": atletic_woman_data,
        },
    )


@app.get("/curvywoman", response_class=HTMLResponse)
async def curvywoman(request: Request) -> HTMLResponse:
    curvy_woman_data = json.load(open("json_templates_in/curvy_woman.json"))
    return templates.TemplateResponse(
        "curvywoman.html",
        {
            "request": request,
            "curvy_woman_data": curvy_woman_data,
        },
    )


@app.get("/scarfwoman", response_class=HTMLResponse)
async def scarfwoman(request: Request) -> HTMLResponse:
    scarf_woman_data = json.load(open("json_templates_in/scarf_woman.json"))
    return templates.TemplateResponse(
        "scarfwoman.html",
        {
            "request": request,
            "scarf_woman_data": scarf_woman_data,
        },
    )


@app.get("/domesticwoman", response_class=HTMLResponse)
async def domesticwoman(request: Request) -> HTMLResponse:
    domestic_woman_data = json.load(open("json_templates_in/domestic_woman.json"))
    return templates.TemplateResponse(
        "domesticwoman.html",
        {
            "request": request,
            "domestic_woman_data": domestic_woman_data,
        },
    )


@app.get("/gingerwoman", response_class=HTMLResponse)
async def gingerwoman(request: Request) -> HTMLResponse:
    ginger_blonde_data = json.load(open("json_templates_in/ginger_blonde.json"))
    return templates.TemplateResponse(
        "gingerwoman.html",
        {
            "request": request,
            "ginger_blonde_data": ginger_blonde_data,
        },
    )


@app.get("/stringbikini", response_class=HTMLResponse)
async def stringbikini(request: Request) -> HTMLResponse:
    string_bikini_data = json.load(open("json_templates_in/string_bikini.json"))
    return templates.TemplateResponse(
        "stringbikini.html",
        {
            "request": request,
            "string_bikini_data": string_bikini_data,
        },
    )


@app.get("/glamorouswestern", response_class=HTMLResponse)
async def glamorouswestern(request: Request) -> HTMLResponse:
    glamorous_western_data = json.load(open("json_templates_in/glamorous_western.json"))
    return templates.TemplateResponse(
        "glamorouswestern.html",
        {
            "request": request,
            "glamorous_western_data": glamorous_western_data,
        },
    )


@app.get("/onsenwoman", response_class=HTMLResponse)
async def onsenwoman(request: Request) -> HTMLResponse:
    onsen_woman_data = json.load(open("json_templates_in/onsen_woman.json"))
    return templates.TemplateResponse(
        "onsenwoman.html",
        {
            "request": request,
            "onsen_woman_data": onsen_woman_data,
        },
    )


@app.get("/spiderwomans", response_class=HTMLResponse)
async def spiderwomans(request: Request) -> HTMLResponse:
    spider_womans_data = json.load(open("json_templates_in/spider_womans.json"))
    return templates.TemplateResponse(
        "spiderwomans.html",
        {
            "request": request,
            "spider_womans_data": spider_womans_data,
        },
    )


@app.get("/luxuriousbathroom", response_class=HTMLResponse)
async def luxuriousbathroom(request: Request) -> HTMLResponse:
    luxurious_bathroom_data = json.load(open("json_templates_in/luxurious_bathroom.json"))
    return templates.TemplateResponse(
        "luxuriousbathroom.html",
        {
            "request": request,
            "luxurious_bathroom_data": luxurious_bathroom_data,
        },
    )


@app.get("/americangirl", response_class=HTMLResponse)
async def americangirl(request: Request) -> HTMLResponse:
    american_girl_data = json.load(open("json_templates_in/american_girl.json"))
    return templates.TemplateResponse(
        "americangirl.html",
        {
            "request": request,
            "american_girl_data": american_girl_data,
        },
    )


@app.get("/redlatex", response_class=HTMLResponse)
async def redlatex(request: Request) -> HTMLResponse:
    red_latex_data = json.load(open("json_templates_in/red_latex.json"))
    return templates.TemplateResponse(
        "redlatex.html",
        {
            "request": request,
            "red_latex_data": red_latex_data,
        },
    )


@app.get("/blackbob", response_class=HTMLResponse)
async def blackbob(request: Request) -> HTMLResponse:
    black_bob_data = json.load(open("json_templates_in/black_bob.json"))
    return templates.TemplateResponse(
        "blackbob.html",
        {
            "request": request,
            "black_bob_data": black_bob_data,
        },
    )


@app.get("/monokiniwoman", response_class=HTMLResponse)
async def monokiniwoman(request: Request) -> HTMLResponse:
    monokini_woman_data = json.load(open("json_templates_in/monokini_woman.json"))
    return templates.TemplateResponse(
        "monokiniwoman.html",
        {
            "request": request,
            "monokini_woman_data": monokini_woman_data,
        },
    )


@app.get("/tannedwoman", response_class=HTMLResponse)
async def tannedwoman(request: Request) -> HTMLResponse:
    tanned_woman_data = json.load(open("json_templates_in/tanned_woman.json"))
    return templates.TemplateResponse(
        "tannedwoman.html",
        {
            "request": request,
            "tanned_woman_data": tanned_woman_data,
        },
    )


@app.get("/hotelwoman", response_class=HTMLResponse)
async def hotelwoman(request: Request) -> HTMLResponse:
    hotel_woman_data = json.load(open("json_templates_in/hotel_woman.json"))
    return templates.TemplateResponse(
        "hotelwoman.html",
        {
            "request": request,
            "hotel_woman_data": hotel_woman_data,
        },
    )


@app.get("/blueswimming", response_class=HTMLResponse)
async def blueswimming(request: Request) -> HTMLResponse:
    blue_swimming_data = json.load(open("json_templates_in/blue_swimming.json"))
    return templates.TemplateResponse(
        "blueswimming.html",
        {
            "request": request,
            "blue_swimming_data": blue_swimming_data,
        },
    )


@app.get("/gamingwoman", response_class=HTMLResponse)
async def gamingwoman(request: Request) -> HTMLResponse:
    gaming_woman_data = json.load(open("json_templates_in/gaming_woman.json"))
    return templates.TemplateResponse(
        "gamingwoman.html",
        {
            "request": request,
            "gaming_woman_data": gaming_woman_data,
        },
    )


@app.get("/beachportrait", response_class=HTMLResponse)
async def beachportrait(request: Request) -> HTMLResponse:
    beach_portrait_data = json.load(open("json_templates_in/beach_portrait.json"))
    return templates.TemplateResponse(
        "beachportrait.html",
        {
            "request": request,
            "beach_portrait_data": beach_portrait_data,
        },
    )


@app.get("/luxurywomancar", response_class=HTMLResponse)
async def luxurywomancar(request: Request) -> HTMLResponse:
    luxury_woman_car_data = json.load(open("json_templates_in/luxury_woman_car.json"))
    return templates.TemplateResponse(
        "luxurywomancar.html",
        {
            "request": request,
            "luxury_woman_car_data": luxury_woman_car_data,
        },
    )


@app.get("/bluewinter", response_class=HTMLResponse)
async def bluewinter(request: Request) -> HTMLResponse:
    blue_winter_data = json.load(open("json_templates_in/blue_winter.json"))
    return templates.TemplateResponse(
        "bluewinter.html",
        {
            "request": request,
            "blue_winter_data": blue_winter_data,
        },
    )


@app.get("/surfwoman", response_class=HTMLResponse)
async def surfwoman(request: Request) -> HTMLResponse:
    surf_woman_data = json.load(open("json_templates_in/surf_woman.json"))
    return templates.TemplateResponse(
        "surfwoman.html",
        {
            "request": request,
            "surf_woman_data": surf_woman_data,
        },
    )


@app.get("/cartoonwoman", response_class=HTMLResponse)
async def cartoonwoman(request: Request) -> HTMLResponse:
    cartoon_woman_data = json.load(open("json_templates_in/cartoon_woman.json"))
    return templates.TemplateResponse(
        "cartoonwoman.html",
        {
            "request": request,
            "cartoon_woman_data": cartoon_woman_data,
        },
    )


@app.get("/tightfittingwoman", response_class=HTMLResponse)
async def tightfittingwoman(request: Request) -> HTMLResponse:
    tight_fitting_woman_data = json.load(open("json_templates_in/tight_fitting_woman.json"))
    return templates.TemplateResponse(
        "tightfittingwoman.html",
        {
            "request": request,
            "tight_fitting_woman_data": tight_fitting_woman_data,
        },
    )


@app.get("/fitnesswoman", response_class=HTMLResponse)
async def fitnesswoman(request: Request) -> HTMLResponse:
    fitness_woman_data = json.load(open("json_templates_in/fitness_woman.json"))
    return templates.TemplateResponse(
        "fitnesswoman.html",
        {
            "request": request,
            "fitness_woman_data": fitness_woman_data,
        },
    )


@app.get("/womanflower", response_class=HTMLResponse)
async def womanflower(request: Request) -> HTMLResponse:
    woman_flower_data = json.load(open("json_templates_in/woman_flower.json"))
    return templates.TemplateResponse(
        "womanflower.html",
        {
            "request": request,
            "woman_flower_data": woman_flower_data,
        },
    )


@app.get("/purplewoman", response_class=HTMLResponse)
async def purplewoman(request: Request) -> HTMLResponse:
    purple_woman_data = json.load(open("json_templates_in/purple_woman.json"))
    return templates.TemplateResponse(
        "purplewoman.html",
        {
            "request": request,
            "purple_woman_data": purple_woman_data,
        },
    )


@app.get("/candidoutdoor", response_class=HTMLResponse)
async def candidoutdoor(request: Request) -> HTMLResponse:
    candid_outdoor_data = json.load(open("json_templates_in/candid_outdoor.json"))
    return templates.TemplateResponse(
        "candidoutdoor.html",
        {
            "request": request,
            "candid_outdoor_data": candid_outdoor_data,
        },
    )


@app.get("/sunlightred", response_class=HTMLResponse)
async def sunlightred(request: Request) -> HTMLResponse:
    sunlight_red_data = json.load(open("json_templates_in/sunlight_red.json"))
    return templates.TemplateResponse(
        "sunlightred.html",
        {
            "request": request,
            "sunlight_red_data": sunlight_red_data,
        },
    )


@app.get("/cyberwoman", response_class=HTMLResponse)
async def cyberwoman(request: Request) -> HTMLResponse:
    cyber_woman_data = json.load(open("json_templates_in/cyber_woman.json"))
    return templates.TemplateResponse(
        "cyberwoman.html",
        {
            "request": request,
            "cyber_woman_data": cyber_woman_data,
        },
    )


@app.get("/bugattiwoman", response_class=HTMLResponse)
async def bugattiwoman(request: Request) -> HTMLResponse:
    bugatti_woman_data = json.load(open("json_templates_in/bugatti_woman.json"))
    return templates.TemplateResponse(
        "bugattiwoman.html",
        {
            "request": request,
            "bugatti_woman_data": bugatti_woman_data,
        },
    )


@app.get("/skyscraperwoman", response_class=HTMLResponse)
async def skyscraperwoman(request: Request) -> HTMLResponse:
    skyscraper_woman_data = json.load(open("json_templates_in/skyscraper_woman.json"))
    return templates.TemplateResponse(
        "skyscraperwoman.html",
        {
            "request": request,
            "skyscraper_woman_data": skyscraper_woman_data,
        },
    )


@app.get("/lazywoman", response_class=HTMLResponse)
async def lazywoman(request: Request) -> HTMLResponse:
    lazy_woman_data = json.load(open("json_templates_in/lazy_woman.json"))
    return templates.TemplateResponse(
        "lazywoman.html",
        {
            "request": request,
            "lazy_woman_data": lazy_woman_data,
        },
    )


@app.get("/asiaduowoman", response_class=HTMLResponse)
async def asiaduowoman(request: Request) -> HTMLResponse:
    asia_duo_woman_data = json.load(open("json_templates_in/asia_duo_woman.json"))
    return templates.TemplateResponse(
        "asiaduowoman.html",
        {
            "request": request,
            "asia_duo_woman_data": asia_duo_woman_data,
        },
    )


@app.get("/trianglebikini", response_class=HTMLResponse)
async def trianglebikini(request: Request) -> HTMLResponse:
    triangle_bikini_data = json.load(open("json_templates_in/triangle_bikini.json"))
    return templates.TemplateResponse(
        "trianglebikini.html",
        {
            "request": request,
            "triangle_bikini_data": triangle_bikini_data,
        },
    )


@app.get("/anadearma", response_class=HTMLResponse)
async def anadearma(request: Request) -> HTMLResponse:
    ana_de_arma_data = json.load(open("json_templates_in/ana_de_arma.json"))
    return templates.TemplateResponse(
        "anadearma.html",
        {
            "request": request,
            "ana_de_arma_data": ana_de_arma_data,
        },
    )


@app.post("/privatepoolnight", response_class=HTMLResponse)
async def privatepoolnight_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/private_pool_night.json")
    private_pool_night_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    list_fields = {
        "scene.environment",
        "lighting.effect",
        "realism_constraints",
    }

    for key, value in form.items():
        if value is None:
            continue
        if key.startswith("subjects.0."):
            subject_path = key[len("subjects.0."):]
            subjects = private_pool_night_data.get("subjects")
            if not isinstance(subjects, list) or not subjects:
                subjects = [{}]
                private_pool_night_data["subjects"] = subjects
            if key in list_fields:
                _set_nested_value(subjects[0], subject_path, _parse_list(str(value)))
            else:
                _set_nested_value(subjects[0], subject_path, str(value))
            continue
        if key in list_fields:
            _set_nested_value(private_pool_night_data, key, _parse_list(str(value)))
        else:
            _set_nested_value(private_pool_night_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/privatepoolnight.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=private_pool_night_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(f"/static/images_generated/{Path(result_path).name}")
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "privatepoolnight.html",
        {
            "request": request,
            "private_pool_night_data": private_pool_night_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/winterfashion", response_class=HTMLResponse)
async def winterfashion_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/winter_fashion.json")
    winter_fashion_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    for key, value in form.items():
        if value is None:
            continue
        _set_nested_value(winter_fashion_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/winterfashion.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=winter_fashion_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "winterfashion.html",
        {
            "request": request,
            "winter_fashion_data": winter_fashion_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/analogmirror", response_class=HTMLResponse)
async def analogmirror_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/analog_mirror.json")
    analog_mirror_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    for key, value in form.items():
        if value is None:
            continue
        _set_nested_value(analog_mirror_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/analogmirror.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=analog_mirror_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "analogmirror.html",
        {
            "request": request,
            "analog_mirror_data": analog_mirror_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/smallbedroom", response_class=HTMLResponse)
async def smallbedroom_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/small_bedroom.json")
    small_bedroom_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    list_fields = {
        "scene.environment",
        "lighting.behavior",
        "iphone_rules",
    }

    for key, value in form.items():
        if value is None:
            continue
        if key in list_fields:
            _set_nested_value(small_bedroom_data, key, _parse_list(str(value)))
        else:
            _set_nested_value(small_bedroom_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/smallbedroom.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=small_bedroom_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "smallbedroom.html",
        {
            "request": request,
            "small_bedroom_data": small_bedroom_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/rusticbedroom", response_class=HTMLResponse)
async def rusticbedroom_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/rustic_bedroom.json")
    rustic_bedroom_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    list_fields = {
        "prompt_data.quality_tags",
    }

    for key, value in form.items():
        if value is None:
            continue
        if key in list_fields:
            _set_nested_value(rustic_bedroom_data, key, _parse_list(str(value)))
        else:
            _set_nested_value(rustic_bedroom_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/rusticbedroom.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=rustic_bedroom_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "rusticbedroom.html",
        {
            "request": request,
            "rustic_bedroom_data": rustic_bedroom_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/otakucorner", response_class=HTMLResponse)
async def otakucorner_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/otaku_corner.json")
    otaku_corner_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    list_fields = {
        "environment.furnishings",
        "negatives",
    }

    for key, value in form.items():
        if value is None:
            continue
        if key in list_fields:
            _set_nested_value(otaku_corner_data, key, _parse_list(str(value)))
        else:
            _set_nested_value(otaku_corner_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/otakucorner.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=otaku_corner_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "otakucorner.html",
        {
            "request": request,
            "otaku_corner_data": otaku_corner_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/nightportrait", response_class=HTMLResponse)
async def nightportrait_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/night_portrait.json")
    night_portrait_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    for key, value in form.items():
        if value is None:
            continue
        _set_nested_value(night_portrait_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/nightportrait.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=night_portrait_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "nightportrait.html",
        {
            "request": request,
            "night_portrait_data": night_portrait_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/edgywoman", response_class=HTMLResponse)
async def edgywoman_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/edgy_woman.json")
    edgy_woman_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    list_fields = {
        "styling.accessories",
    }

    for key, value in form.items():
        if value is None:
            continue
        if key in list_fields:
            _set_nested_value(edgy_woman_data, key, _parse_list(str(value)))
        else:
            _set_nested_value(edgy_woman_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/edgywoman.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=edgy_woman_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "edgywoman.html",
        {
            "request": request,
            "edgy_woman_data": edgy_woman_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/allurgingwoman", response_class=HTMLResponse)
async def allurgingwoman_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/allurging_woman.json")
    allurging_woman_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    list_fields = {
        "mood",
        "technical_details",
    }

    for key, value in form.items():
        if value is None:
            continue
        if key in list_fields:
            _set_nested_value(allurging_woman_data, key, _parse_list(str(value)))
        else:
            _set_nested_value(allurging_woman_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/allurgingwoman.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=allurging_woman_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "allurgingwoman.html",
        {
            "request": request,
            "allurging_woman_data": allurging_woman_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/minidress", response_class=HTMLResponse)
async def minidress_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/mini_dress.json")
    mini_dress_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    list_fields = {
        "apparel.details",
        "environment.architectural_features",
    }

    for key, value in form.items():
        if value is None:
            continue
        if key in list_fields:
            _set_nested_value(mini_dress_data, key, _parse_list(str(value)))
        else:
            _set_nested_value(mini_dress_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/minidress.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=mini_dress_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "minidress.html",
        {
            "request": request,
            "mini_dress_data": mini_dress_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/whitesend", response_class=HTMLResponse)
async def whitesend_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/white_send.json")
    white_send_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    for key, value in form.items():
        if value is None:
            continue
        _set_nested_value(white_send_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/whitesend.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=white_send_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "whitesend.html",
        {
            "request": request,
            "white_send_data": white_send_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/whitetop", response_class=HTMLResponse)
async def whitetop_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/white_top.json")
    white_top_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    for key, value in form.items():
        if value is None:
            continue
        _set_nested_value(white_top_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/whitetop.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=white_top_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "whitetop.html",
        {
            "request": request,
            "white_top_data": white_top_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/luxuryaesthetic", response_class=HTMLResponse)
async def luxuryaesthetic_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/luxury_aesthetic.json")
    luxury_aesthetic_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    list_fields = {
        "attire.design_details",
        "environment.architecture",
        "environment.decor",
    }

    for key, value in form.items():
        if value is None:
            continue
        if key in list_fields:
            _set_nested_value(luxury_aesthetic_data, key, _parse_list(str(value)))
        else:
            _set_nested_value(luxury_aesthetic_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/luxuryaesthetic.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=luxury_aesthetic_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "luxuryaesthetic.html",
        {
            "request": request,
            "luxury_aesthetic_data": luxury_aesthetic_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/atleticwoman", response_class=HTMLResponse)
async def atleticwoman_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/atletic_woman.json")
    atletic_woman_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    list_fields = {
        "environment.background.elements",
    }

    for key, value in form.items():
        if value is None:
            continue
        if key in list_fields:
            _set_nested_value(atletic_woman_data, key, _parse_list(str(value)))
        else:
            _set_nested_value(atletic_woman_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/atleticwoman.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=atletic_woman_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "atleticwoman.html",
        {
            "request": request,
            "atletic_woman_data": atletic_woman_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/curvywoman", response_class=HTMLResponse)
async def curvywoman_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/curvy_woman.json")
    curvy_woman_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    list_fields = {
        "subject.accessories",
        "framing.focus_progression",
        "environment.objects",
        "visual_emphasis",
    }

    for key, value in form.items():
        if value is None:
            continue
        if key in list_fields:
            _set_nested_value(curvy_woman_data, key, _parse_list(str(value)))
        else:
            _set_nested_value(curvy_woman_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/curvywoman.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=curvy_woman_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "curvywoman.html",
        {
            "request": request,
            "curvy_woman_data": curvy_woman_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/scarfwoman", response_class=HTMLResponse)
async def scarfwoman_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/scarf_woman.json")
    scarf_woman_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    list_fields = {
        "accessories.jewelry",
    }

    for key, value in form.items():
        if value is None:
            continue
        if key in list_fields:
            _set_nested_value(scarf_woman_data, key, _parse_list(str(value)))
        else:
            _set_nested_value(scarf_woman_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/scarfwoman.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=scarf_woman_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "scarfwoman.html",
        {
            "request": request,
            "scarf_woman_data": scarf_woman_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/domesticwoman", response_class=HTMLResponse)
async def domesticwoman_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/domestic_woman.json")
    domestic_woman_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    list_fields = {
        "environment.background_elements.furniture.side_table.visible_items",
    }

    for key, value in form.items():
        if value is None:
            continue
        if key in list_fields:
            _set_nested_value(domestic_woman_data, key, _parse_list(str(value)))
        else:
            _set_nested_value(domestic_woman_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/domesticwoman.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=domestic_woman_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "domesticwoman.html",
        {
            "request": request,
            "domestic_woman_data": domestic_woman_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/gingerwoman", response_class=HTMLResponse)
async def gingerwoman_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/ginger_blonde.json")
    ginger_blonde_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    for key, value in form.items():
        if value is None:
            continue
        _set_nested_value(ginger_blonde_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/gingerwoman.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=ginger_blonde_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "gingerwoman.html",
        {
            "request": request,
            "ginger_blonde_data": ginger_blonde_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/stringbikini", response_class=HTMLResponse)
async def stringbikini_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/string_bikini.json")
    string_bikini_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    for key, value in form.items():
        if value is None:
            continue
        _set_nested_value(string_bikini_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/stringbikini.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=string_bikini_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "stringbikini.html",
        {
            "request": request,
            "string_bikini_data": string_bikini_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/glamorouswestern", response_class=HTMLResponse)
async def glamorouswestern_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/glamorous_western.json")
    glamorous_western_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    list_fields = {
        "fashion_and_apparel.accessories",
    }

    for key, value in form.items():
        if value is None:
            continue
        if key in list_fields:
            _set_nested_value(glamorous_western_data, key, _parse_list(str(value)))
        else:
            _set_nested_value(glamorous_western_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/glamorouswestern.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=glamorous_western_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "glamorouswestern.html",
        {
            "request": request,
            "glamorous_western_data": glamorous_western_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/onsenwoman", response_class=HTMLResponse)
async def onsenwoman_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/onsen_woman.json")
    onsen_woman_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    for key, value in form.items():
        if value is None:
            continue
        _set_nested_value(onsen_woman_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/onsenwoman.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=onsen_woman_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "onsenwoman.html",
        {
            "request": request,
            "onsen_woman_data": onsen_woman_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/spiderwomans", response_class=HTMLResponse)
async def spiderwomans_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/spider_womans.json")
    spider_womans_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    for key, value in form.items():
        if value is None:
            continue
        _set_nested_value(spider_womans_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/spiderwomans.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=spider_womans_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "spiderwomans.html",
        {
            "request": request,
            "spider_womans_data": spider_womans_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/luxuriousbathroom", response_class=HTMLResponse)
async def luxuriousbathroom_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/luxurious_bathroom.json")
    luxurious_bathroom_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    for key, value in form.items():
        if value is None:
            continue
        _set_nested_value(luxurious_bathroom_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/luxuriousbathroom.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=luxurious_bathroom_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "luxuriousbathroom.html",
        {
            "request": request,
            "luxurious_bathroom_data": luxurious_bathroom_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/americangirl", response_class=HTMLResponse)
async def americangirl_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/american_girl.json")
    american_girl_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    list_fields = {
        "wardrobe_and_setting.apparel.details",
        "technical_emulation.lighting.effects",
        "technical_emulation.camera_characteristics.sensor_artifacts",
        "technical_emulation.camera_characteristics.optics",
    }

    for key, value in form.items():
        if value is None:
            continue
        if key in list_fields:
            _set_nested_value(american_girl_data, key, _parse_list(str(value)))
        else:
            _set_nested_value(american_girl_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/americangirl.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=american_girl_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "americangirl.html",
        {
            "request": request,
            "american_girl_data": american_girl_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/redlatex", response_class=HTMLResponse)
async def redlatex_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/red_latex.json")
    red_latex_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    list_fields = {
        "image_specification.environment_and_setting.background_elements",
        "image_specification.constraints.negative_prompt",
    }

    for key, value in form.items():
        if value is None:
            continue
        if key in list_fields:
            _set_nested_value(red_latex_data, key, _parse_list(str(value)))
        else:
            _set_nested_value(red_latex_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/redlatex.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=red_latex_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "redlatex.html",
        {
            "request": request,
            "red_latex_data": red_latex_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/blackbob", response_class=HTMLResponse)
async def blackbob_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/black_bob.json")
    black_bob_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    for key, value in form.items():
        if value is None:
            continue
        _set_nested_value(black_bob_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/blackbob.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=black_bob_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "blackbob.html",
        {
            "request": request,
            "black_bob_data": black_bob_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/monokiniwoman", response_class=HTMLResponse)
async def monokiniwoman_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/monokini_woman.json")
    monokini_woman_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    list_fields = {
        "controlnet.pose_control.constraints",
        "controlnet.depth_control.constraints",
        "negative_prompt.forbidden_elements",
    }

    for key, value in form.items():
        if value is None:
            continue
        if key in list_fields:
            _set_nested_value(monokini_woman_data, key, _parse_list(str(value)))
        else:
            _set_nested_value(monokini_woman_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/monokiniwoman.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=monokini_woman_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "monokiniwoman.html",
        {
            "request": request,
            "monokini_woman_data": monokini_woman_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/tannedwoman", response_class=HTMLResponse)
async def tannedwoman_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/tanned_woman.json")
    tanned_woman_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    list_fields = {
        "environment.elements",
        "controlnet.pose_control.constraints",
        "controlnet.depth_control.constraints",
        "negative_prompt.forbidden_concepts",
    }

    for key, value in form.items():
        if value is None:
            continue
        if key in list_fields:
            _set_nested_value(tanned_woman_data, key, _parse_list(str(value)))
        else:
            _set_nested_value(tanned_woman_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/tannedwoman.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=tanned_woman_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "tannedwoman.html",
        {
            "request": request,
            "tanned_woman_data": tanned_woman_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/hotelwoman", response_class=HTMLResponse)
async def hotelwoman_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/hotel_woman.json")
    hotel_woman_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    list_fields = {
        "image_prompt.meta.texture_rules",
        "image_prompt.scene.key_background_elements",
        "image_prompt.scene.foreground",
        "image_prompt.scene.extra_realism_details",
        "image_prompt.lighting.sources",
        "image_prompt.lighting.look",
        "image_prompt.outfit.top.fit_rules",
        "image_prompt.pose.exact_pose_match",
        "image_prompt.pose.micro_motion",
        "image_prompt.camera_composition.realism_notes",
        "image_prompt.negative_prompt",
    }

    for key, value in form.items():
        if value is None:
            continue
        if key in list_fields:
            _set_nested_value(hotel_woman_data, key, _parse_list(str(value)))
        else:
            _set_nested_value(hotel_woman_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/hotelwoman.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=hotel_woman_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "hotelwoman.html",
        {
            "request": request,
            "hotel_woman_data": hotel_woman_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/blueswimming", response_class=HTMLResponse)
async def blueswimming_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/blue_swimming.json")
    blue_swimming_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    list_fields = {
        "scene.environment",
    }

    for key, value in form.items():
        if value is None:
            continue
        if key in list_fields:
            _set_nested_value(blue_swimming_data, key, _parse_list(str(value)))
        else:
            _set_nested_value(blue_swimming_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/blueswimming.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=blue_swimming_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "blueswimming.html",
        {
            "request": request,
            "blue_swimming_data": blue_swimming_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/gamingwoman", response_class=HTMLResponse)
async def gamingwoman_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/gaming_woman.json")
    gaming_woman_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    for key, value in form.items():
        if value is None:
            continue
        _set_nested_value(gaming_woman_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/gamingwoman.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=gaming_woman_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "gamingwoman.html",
        {
            "request": request,
            "gaming_woman_data": gaming_woman_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/beachportrait", response_class=HTMLResponse)
async def beachportrait_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/beach_portrait.json")
    beach_portrait_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    for key, value in form.items():
        if value is None:
            continue
        _set_nested_value(beach_portrait_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/beachportrait.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=beach_portrait_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "beachportrait.html",
        {
            "request": request,
            "beach_portrait_data": beach_portrait_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/luxurywomancar", response_class=HTMLResponse)
async def luxurywomancar_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/luxury_woman_car.json")
    luxury_woman_car_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    for key, value in form.items():
        if value is None:
            continue
        _set_nested_value(luxury_woman_car_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/luxurywomancar.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=luxury_woman_car_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "luxurywomancar.html",
        {
            "request": request,
            "luxury_woman_car_data": luxury_woman_car_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/bluewinter", response_class=HTMLResponse)
async def bluewinter_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/blue_winter.json")
    blue_winter_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    list_fields = {
        "environment.background_elements",
    }

    for key, value in form.items():
        if value is None:
            continue
        if key in list_fields:
            _set_nested_value(blue_winter_data, key, _parse_list(str(value)))
        else:
            _set_nested_value(blue_winter_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/bluewinter.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=blue_winter_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "bluewinter.html",
        {
            "request": request,
            "blue_winter_data": blue_winter_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/surfwoman", response_class=HTMLResponse)
async def surfwoman_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/surf_woman.json")
    surf_woman_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    list_fields = {
        "environment.background_elements",
    }

    for key, value in form.items():
        if value is None:
            continue
        if key in list_fields:
            _set_nested_value(surf_woman_data, key, _parse_list(str(value)))
        else:
            _set_nested_value(surf_woman_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/surfwoman.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=surf_woman_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "surfwoman.html",
        {
            "request": request,
            "surf_woman_data": surf_woman_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/cartoonwoman", response_class=HTMLResponse)
async def cartoonwoman_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/cartoon_woman.json")
    cartoon_woman_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    list_fields = {
        "environment.background_elements",
    }

    for key, value in form.items():
        if value is None:
            continue
        if key in list_fields:
            _set_nested_value(cartoon_woman_data, key, _parse_list(str(value)))
        else:
            _set_nested_value(cartoon_woman_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/cartoonwoman.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=cartoon_woman_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "cartoonwoman.html",
        {
            "request": request,
            "cartoon_woman_data": cartoon_woman_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/tightfittingwoman", response_class=HTMLResponse)
async def tightfittingwoman_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/tight_fitting_woman.json")
    tight_fitting_woman_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    list_fields = {
        "technical_modifiers",
    }

    for key, value in form.items():
        if value is None:
            continue
        if key in list_fields:
            _set_nested_value(tight_fitting_woman_data, key, _parse_list(str(value)))
        else:
            _set_nested_value(tight_fitting_woman_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/tightfittingwoman.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=tight_fitting_woman_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "tightfittingwoman.html",
        {
            "request": request,
            "tight_fitting_woman_data": tight_fitting_woman_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/fitnesswoman", response_class=HTMLResponse)
async def fitnesswoman_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/fitness_woman.json")
    fitness_woman_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    list_fields = {
        "style",
        "environment.background_elements",
        "quality_tags",
    }

    for key, value in form.items():
        if value is None:
            continue
        if key in list_fields:
            _set_nested_value(fitness_woman_data, key, _parse_list(str(value)))
        else:
            _set_nested_value(fitness_woman_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/fitnesswoman.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=fitness_woman_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "fitnesswoman.html",
        {
            "request": request,
            "fitness_woman_data": fitness_woman_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/womanflower", response_class=HTMLResponse)
async def womanflower_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/woman_flower.json")
    woman_flower_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    list_fields = {
        "scene.environment",
    }

    for key, value in form.items():
        if value is None:
            continue
        if key in list_fields:
            _set_nested_value(woman_flower_data, key, _parse_list(str(value)))
        else:
            _set_nested_value(woman_flower_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/womanflower.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=woman_flower_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "womanflower.html",
        {
            "request": request,
            "woman_flower_data": woman_flower_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/purplewoman", response_class=HTMLResponse)
async def purplewoman_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/purple_woman.json")
    purple_woman_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    for key, value in form.items():
        if value is None:
            continue
        _set_nested_value(purple_woman_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/purplewoman.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=purple_woman_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "purplewoman.html",
        {
            "request": request,
            "purple_woman_data": purple_woman_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/candidoutdoor", response_class=HTMLResponse)
async def candidoutdoor_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/candid_outdoor.json")
    candid_outdoor_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    list_fields = {
        "scene.environment",
    }

    for key, value in form.items():
        if value is None:
            continue
        if key in list_fields:
            _set_nested_value(candid_outdoor_data, key, _parse_list(str(value)))
        else:
            _set_nested_value(candid_outdoor_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/candidoutdoor.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=candid_outdoor_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "candidoutdoor.html",
        {
            "request": request,
            "candid_outdoor_data": candid_outdoor_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/sunlightred", response_class=HTMLResponse)
async def sunlightred_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/sunlight_red.json")
    sunlight_red_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    list_fields = {
        "scene.environment",
    }

    for key, value in form.items():
        if value is None:
            continue
        if key in list_fields:
            _set_nested_value(sunlight_red_data, key, _parse_list(str(value)))
        else:
            _set_nested_value(sunlight_red_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/sunlightred.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=sunlight_red_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "sunlightred.html",
        {
            "request": request,
            "sunlight_red_data": sunlight_red_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/cyberwoman", response_class=HTMLResponse)
async def cyberwoman_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/cyber_woman.json")
    cyber_woman_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    for key, value in form.items():
        if value is None:
            continue
        _set_nested_value(cyber_woman_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/cyberwoman.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=cyber_woman_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "cyberwoman.html",
        {
            "request": request,
            "cyber_woman_data": cyber_woman_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/bugattiwoman", response_class=HTMLResponse)
async def bugattiwoman_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/bugatti_woman.json")
    bugatti_woman_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    for key, value in form.items():
        if value is None:
            continue
        _set_nested_value(bugatti_woman_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/bugattiwoman.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=bugatti_woman_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "bugattiwoman.html",
        {
            "request": request,
            "bugatti_woman_data": bugatti_woman_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/skyscraperwoman", response_class=HTMLResponse)
async def skyscraperwoman_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/skyscraper_woman.json")
    skyscraper_woman_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    for key, value in form.items():
        if value is None:
            continue
        _set_nested_value(skyscraper_woman_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/skyscraperwoman.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=skyscraper_woman_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "skyscraperwoman.html",
        {
            "request": request,
            "skyscraper_woman_data": skyscraper_woman_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/lazywoman", response_class=HTMLResponse)
async def lazywoman_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/lazy_woman.json")
    lazy_woman_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    for key, value in form.items():
        if value is None:
            continue
        _set_nested_value(lazy_woman_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/lazywoman.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=lazy_woman_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "lazywoman.html",
        {
            "request": request,
            "lazy_woman_data": lazy_woman_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/asiaduowoman", response_class=HTMLResponse)
async def asiaduowoman_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/asia_duo_woman.json")
    asia_duo_woman_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    for key, value in form.items():
        if value is None:
            continue
        _set_nested_value(asia_duo_woman_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/asiaduowoman.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=asia_duo_woman_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "asiaduowoman.html",
        {
            "request": request,
            "asia_duo_woman_data": asia_duo_woman_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/trianglebikini", response_class=HTMLResponse)
async def trianglebikini_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/triangle_bikini.json")
    triangle_bikini_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    for key, value in form.items():
        if value is None:
            continue
        _set_nested_value(triangle_bikini_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/trianglebikini.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=triangle_bikini_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "trianglebikini.html",
        {
            "request": request,
            "triangle_bikini_data": triangle_bikini_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )


@app.post("/anadearma", response_class=HTMLResponse)
async def anadearma_submit(request: Request) -> HTMLResponse:
    form = await request.form()

    base_path = Path("json_templates_in/ana_de_arma.json")
    ana_de_arma_data = json.loads(base_path.read_text(encoding="utf-8"))
    generation_status: Optional[str] = None
    generation_error: Optional[str] = None
    generated_image_path: Optional[str] = None

    for key, value in form.items():
        if value is None:
            continue
        _set_nested_value(ana_de_arma_data, key, str(value))

    output_image_path = f"{IMAGES_GENERATED}/anadearma.png"

    try:
        result_path = generate_variation_from_template_json(
            template_json=ana_de_arma_data,
            out_image_path=output_image_path,
        )
        generation_status = "completed"
        generated_image_path = _with_cache_buster(
            f"/static/images_generated/{Path(result_path).name}"
        )
    except Exception as exc:
        generation_status = "failed"
        generation_error = str(exc)

    return templates.TemplateResponse(
        "anadearma.html",
        {
            "request": request,
            "ana_de_arma_data": ana_de_arma_data,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "generated_image_path": generated_image_path,
        },
    )
## Nano Banana Templates Ends

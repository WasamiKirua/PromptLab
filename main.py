from __future__ import annotations

import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, File, Form, Request, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from captioning_utils import generate_caption_and_prompt_from_image, generate_prompt_from_text

load_dotenv()

BASE_DIR = Path(__file__).parent
ENV_PATH = BASE_DIR / ".env"
TEMPLATES_DIR = BASE_DIR / "templates"

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

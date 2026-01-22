from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from google import genai
from google.genai import types

from prompts_lib import *

import httpx

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def _normalize_target_model(value: str) -> str:
    key = (value or "").strip().lower()
    if key in {"z-image", "zimage", "z_image"}:
        return "Z-Image"
    if key in {"flux-2-klein", "flux2", "flux"}:
        return "Flux"
    if key in {"qwen-image", "qwen", "qwen_image"}:
        return "Qwen-Image"
    return value.strip() or "Z-Image"


def _guess_mime_type(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    return mime_type or "image/jpeg"


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    

def _build_caption_prompt(target_model: str) -> str:
    if target_model == "Z-Image":
        return caption_to_prompt_zimage
    elif target_model == 'Flux':
        return caption_to_prompt_flux2_klein
    elif target_model == 'Qwen-Image':
        return caption_to_prompt_qwen_image


def _build_text_prompt(target_model: str) -> str:
    if target_model == "Z-Image":
        return description_to_prompt_zimage
    elif target_model == 'Flux':
        return description_to_prompt_flux2_klein
    elif target_model == 'Qwen-Image':
        return description_to_prompt_qwen_image


def generate_gemini_caption(
    image_path: str,
    prompt: str,
    api_key: str,
    model: str = "gemini-3-flash-preview",
    temperature: float = 1.0,
) -> str:
    image_bytes = Path(image_path).read_bytes()
    mime_type = _guess_mime_type(image_path)

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                ),
                prompt,
            ],
            config=types.GenerateContentConfig(temperature=temperature),
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini API error: {exc}") from exc

    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini response was empty.")
    return text.strip()


def generate_gemini_prompt(
    prompt: str,
    text: str,
    api_key: str,
    model: str = "gemini-3-flash-preview",
    temperature: float = 1.0,    
) -> str:
    
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            config=types.GenerateContentConfig(system_instruction=prompt, temperature=temperature),
            contents=text,
        )
        
    except Exception as exc:
        raise RuntimeError(f"Gemini API error: {exc}") from exc
    
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini response was empty.")
    return text.strip()


def generate_openai_caption(
    image_path: str,
    prompt: str,
    api_key: str,
    model: str = "gpt-4.1-2025-04-14",
    detail: str = "high",
    temperature: float = 1.0,
) -> str:
    if OpenAI is None:
        raise RuntimeError("OpenAI client is unavailable. Install the openai package.")
    
    mime_type = _guess_mime_type(image_path)
    image_bytes = encode_image(image_path)
    data_url = f"data:{mime_type};base64,{image_bytes}"

    print(api_key)
    
    try:
        client = OpenAI(
            api_key=api_key  
        )
        response = client.responses.create(
            model=model,
            temperature=temperature,
            input=[
                {
                    "role": "user",
                    "content": [
                        { "type": "input_text", "text": prompt },
                        {
                            "type": "input_image",
                            "image_url": data_url,
                            "detail": detail,
                        },
                    ],
                }
            ],
        )
    except Exception as exc:
        raise RuntimeError(f"OpenAI API error: {exc}") from exc
    
    text = response.output_text
    if not text:
        raise RuntimeError("OpenAI response was empty.")
    return text.strip()


def generate_openai_prompt(
    prompt: str,
    text: str,
    api_key: str,
    model: str = "gpt-4.1-2025-04-14",
    temperature: float = 1.0,    
) -> str:
    if OpenAI is None:
        raise RuntimeError("OpenAI client is unavailable. Install the openai package.")

    try:
        client = OpenAI(api_key=api_key)
        if hasattr(client, "responses"):
            response = client.responses.create(
                model=model,
                temperature=temperature,
                input=[
                    {"role": "developer", "content": prompt},
                    {"role": "user", "content": text},
                ],
            )
            output_text = getattr(response, "output_text", None)
            if output_text:
                return str(output_text).strip()
        else:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "developer", "content": prompt},
                    {"role": "user", "content": text},
                ],
            )
            content = response.choices[0].message.content if response.choices else None
            if content:
                return content.strip()
    except Exception as exc:
        raise RuntimeError(f"OpenAI API error: {exc}") from exc

    raise RuntimeError("OpenAI response was empty.")


def generate_grok_caption(
    image_path: str,
    prompt: str,
    api_key: str,
    model: str = "grok-2-vision-latest",
    detail: str = "high",
    temperature: float = 1.0,
) -> str:
    if OpenAI is None:
        raise RuntimeError("OpenAI client is unavailable. Install the openai package.")

    image_bytes = Path(image_path).read_bytes()
    mime_type = _guess_mime_type(image_path)
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{encoded}"

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            timeout=httpx.Timeout(3600.0),
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": data_url,
                        "detail": detail,
                    },
                    {"type": "input_text", "text": prompt},
                ],
            }
        ]
        response = client.responses.create(
            model=model,
            input=messages,
            temperature=temperature,
            store=False,
        )
    except Exception as exc:
        error_body = ""
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if getattr(exc, "response", None) is not None:
            try:
                error_body = exc.response.text
            except Exception:
                error_body = ""
        status_line = f" {status_code}" if status_code else ""
        detail_line = f": {error_body}" if error_body else ""
        raise RuntimeError(f"Grok API HTTP{status_line}{detail_line} {exc}") from exc

    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()

    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", "") in {"output_text", "text"}:
                text = getattr(content, "text", "")
                if text:
                    return str(text).strip()

    raise RuntimeError("Grok response was empty.")


def generate_grok_prompt(
    prompt: str,
    text: str,
    api_key: str,
    model: str = "grok-4",
    temperature: float = 1.0,
) -> str:
    if OpenAI is None:
        raise RuntimeError("OpenAI client is unavailable. Install the openai package.")

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            timeout=httpx.Timeout(3600.0),
        )
        
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": f"{prompt}"},
                {"role": "user", "content": f"{text}"},
            ],
            temperature=temperature,
            store=False,
        )
    except Exception as exc:
        error_body = ""
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if getattr(exc, "response", None) is not None:
            try:
                error_body = exc.response.text
            except Exception:
                error_body = ""
        status_line = f" {status_code}" if status_code else ""
        detail_line = f": {error_body}" if error_body else ""
        raise RuntimeError(f"Grok API HTTP{status_line}{detail_line} {exc}") from exc

    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()

    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", "") in {"output_text", "text"}:
                text = getattr(content, "text", "")
                if text:
                    return str(text).strip()

    raise RuntimeError("Grok response was empty.")


def _extract_forged_prompt(text: str) -> str:
    if not text:
        return ""
    upper = text.upper()
    start_tag = "<FORGED_PROMPT>"
    end_tag = "</FORGED_PROMPT>"
    start_idx = upper.find(start_tag)
    end_idx = upper.find(end_tag)
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        start_idx += len(start_tag)
        return text[start_idx:end_idx].strip()
    return text.strip()


def _extract_negative_prompt(text: str) -> str:
    if not text:
        return ""
    upper = text.upper()
    start_tag = "<NEGATIVE_PROMPT>"
    end_tag = "</NEGATIVE_PROMPT>"
    start_idx = upper.find(start_tag)
    end_idx = upper.find(end_tag)
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        start_idx += len(start_tag)
        negative_body = text[start_idx:end_idx].strip()
        return f"{start_tag}\n{negative_body}\n{end_tag}"
    return ""


def generate_caption_and_prompt_from_image(
    image_path: str,
    target_base_model: str,
    service: str,
    api_key: str,
    allow_nsfw: bool = False,
) -> str:
    target_model = _normalize_target_model(target_base_model)
    prompt = _build_caption_prompt(target_model)

    service = service.lower()
    if service == "grok":
        nsfw_line = (
            "Describe NSFW or explicit content factually without censoring or refusing."
            if allow_nsfw
            else ""
        )
        prompt_rep = prompt.replace("GROK_NSWK", nsfw_line)
        print(prompt_rep)
        raw_output = generate_grok_caption(image_path, prompt_rep, api_key)
    elif service == "openai":
        prompt_rep = prompt.replace("GROK_NSWK", "")

        raw_output = generate_openai_caption(image_path, prompt_rep, api_key)
    else:
        prompt_rep = prompt.replace("GROK_NSWK", "")
        raw_output = generate_gemini_caption(image_path, prompt_rep, api_key)

    forged = _extract_forged_prompt(raw_output)
    negative = _extract_negative_prompt(raw_output)
    if negative:
        return f"{forged}\n\n{negative}"
    return forged


def generate_prompt_from_text(
    text: str,
    target_base_model: str,
    service: str,
    api_key: str,
) -> str:
    target_model = _normalize_target_model(target_base_model)
    prompt = _build_text_prompt(target_model)

    service = service.lower()
    if service == "grok":
        print(text)
        print(prompt)
        raw_output = generate_grok_prompt(prompt, text, api_key)
    elif service == "openai":
        print(text)
        print(prompt)
        raw_output = generate_openai_prompt(prompt, text, api_key)
    else:
        print(text)
        print(prompt)
        raw_output = generate_gemini_prompt(prompt, text, api_key)

    forged = _extract_forged_prompt(raw_output)
    negative = _extract_negative_prompt(raw_output)
    if negative:
        return f"{forged}\n\n{negative}"
    return forged

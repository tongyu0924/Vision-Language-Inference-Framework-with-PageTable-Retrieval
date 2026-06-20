import io
import re
import time
import base64
import hashlib
import urllib.parse
from typing import Optional, Union, List, Dict
import requests
from PIL import Image

from config import client, WIKI_USER_AGENT


def resize_image_for_api(image: Image.Image, max_side: int = 1024) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        image = image.resize((int(w * scale), int(h * scale)))
    return image


def pil_to_data_url(image: Image.Image) -> str:
    image = resize_image_for_api(image)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def extract_filename_from_wikimedia_url(url: str) -> Optional[str]:
    if not isinstance(url, str):
        return None
    url = url.strip()
    path = urllib.parse.urlparse(url).path
    parts = path.split("/")
    m = re.search(r"/wiki/File:(.+)$", url)
    if m:
        return urllib.parse.unquote(m.group(1))
    if "/thumb/" in path:
        image_like = [
            urllib.parse.unquote(p)
            for p in parts
            if re.search(r"\.(jpg|jpeg|png|webp|gif|tif|tiff|svg)$", p, flags=re.I)
        ]
        if image_like:
            candidate = image_like[-2] if len(image_like) >= 2 else image_like[-1]
            return re.sub(r"^\d+px-", "", candidate)
    if "upload.wikimedia.org" in url:
        last = parts[-1]
        if last:
            return re.sub(r"^\d+px-", "", urllib.parse.unquote(last))
    return None


def commons_file_url_from_filename(filename: str) -> str:
    filename = re.sub(r"^\d+px-", "", str(filename).strip()).replace(" ", "_")
    encoded = urllib.parse.quote(filename, safe="()_'!.,-")
    return f"https://commons.wikimedia.org/wiki/Special:FilePath/{encoded}"


def commons_url_from_filename_hash(filename: str) -> str:
    base = re.sub(r"^\d+px-", "", filename).replace(" ", "_")
    h = hashlib.md5(base.encode("utf-8")).hexdigest()
    if filename != base:
        size_match = re.match(r"^(\d+)px-", filename)
        width = size_match.group(1) if size_match else "512"
        return (f"https://upload.wikimedia.org/wikipedia/commons/thumb/"
               f"{h[0]}/{h[0:2]}/{base}/{width}px-{base}")
    return f"https://upload.wikimedia.org/wikipedia/commons/{h[0]}/{h[0:2]}/{base}"


def wikimedia_url_candidates(url_or_filename: str) -> List[str]:
    candidates = []
    if not isinstance(url_or_filename, str):
        return candidates
    s = url_or_filename.strip()
    if not s:
        return candidates
    if s.startswith("http://"):
        s = "https://" + s[len("http://"):]
    if s.startswith("https://"):
        candidates.append(s)
    fname = extract_filename_from_wikimedia_url(s)
    if fname:
        candidates.append(commons_file_url_from_filename(fname))
        candidates.append(commons_url_from_filename_hash(fname))
    if not s.startswith("http"):
        candidates.append(commons_file_url_from_filename(s))
        candidates.append(commons_url_from_filename_hash(s))
    out, seen = [], set()
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def download_image_to_pil(url_or_filename: str, timeout: int = 30) -> Image.Image:
    candidates = wikimedia_url_candidates(url_or_filename)
    last_error = None
    for url in candidates:
        try:
            r = requests.get(
                url, timeout=timeout,
                headers={"User-Agent": WIKI_USER_AGENT,
                        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8"},
                allow_redirects=True,
            )
            r.raise_for_status()
            ctype = r.headers.get("content-type", "").lower()
            if "svg" in ctype or url.lower().endswith(".svg"):
                raise ValueError(f"SVG not supported: {url}")
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"Could not download image. candidates={candidates}. last_error={repr(last_error)}")


def source_to_openai_image_url(image_source: Union[Image.Image, str]) -> str:
    if isinstance(image_source, Image.Image):
        return pil_to_data_url(image_source)
    if isinstance(image_source, str):
        img = download_image_to_pil(image_source)
        return pil_to_data_url(img)
    raise ValueError("Unsupported image_source type")


def get_viquae_image_source(row: dict, img_key: str):
    candidates = []
    for key in ["url", "image_url", "img_url", "picture_url", "commons_url"]:
        if isinstance(row.get(key), str) and row.get(key).strip():
            candidates.append(row.get(key).strip())
    field = row.get(img_key)
    if isinstance(field, Image.Image):
        return field.convert("RGB")
    if isinstance(field, dict):
        if field.get("bytes"):
            try:
                return Image.open(io.BytesIO(field["bytes"])).convert("RGB")
            except Exception:
                pass
        for k in ["url", "path", "filename", "file_name"]:
            if isinstance(field.get(k), str) and field.get(k).strip():
                candidates.append(field.get(k).strip())
    if isinstance(field, str) and field.strip():
        candidates.append(field.strip())
    for c in candidates:
        if "wikimedia" in c or "commons" in c:
            return c
    return candidates[0] if candidates else None


def call_openai_text(model: str, prompt: str, temperature: float = 0.0,
                     max_tokens: int = 400) -> str:
    for attempt in range(5):
        try:
            resp = client.responses.create(
                model=model,
                input=[{
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }],
                temperature=temperature,
            )
            return resp.output_text
        except Exception as e:
            if attempt == 4:
                raise
            wait = 2 ** attempt
            print(f"[OpenAI text retry] {repr(e)} | sleep {wait}s")
            time.sleep(wait)


def call_openai_vision(model: str, image_source, prompt: str,
                       temperature: float = 0.0) -> str:
    image_url = source_to_openai_image_url(image_source)
    for attempt in range(5):
        try:
            resp = client.responses.create(
                model=model,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }],
                temperature=temperature,
            )
            return resp.output_text
        except Exception as e:
            if attempt == 4:
                raise
            wait = 2 ** attempt
            print(f"[OpenAI vision retry] {repr(e)} | sleep {wait}s")
            time.sleep(wait)

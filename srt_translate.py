#!/usr/bin/env python3
"""Extract, translate, and merge SRT subtitle text with OpenAI-compatible APIs.

The script never sends raw SRT timing metadata to the model. It extracts only
subtitle text into JSONL, translates that text in batches, then merges the
translated text back into the original timeline.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


DEFAULT_BASE_URL = "http://localhost:20128/v1"

DEFAULT_MODEL = "cx/gpt-5.5"
DEFAULT_MODELS = [DEFAULT_MODEL]
DEFAULT_TARGET_CPS = 17.0
DEFAULT_MIN_CUE_CHARS = 24

RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503}
TIMELINE_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}"
)
TIMELINE_DETAIL_RE = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+"
    r"(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
)
TIMELINE_OUTPUT_RE = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+"
    r"(?P<end>\d{2}:\d{2}:\d{2},\d{3})(?P<rest>.*)$"
)
TRAILING_TRANSITION_PHRASES = [
    "Bây giờ chúng ta hãy xem",
    "Bây giờ hãy cùng xem",
    "Chúng ta hãy xem",
    "Giờ hãy cùng xem",
    "Bây giờ hãy xem",
    "Tiếp theo là",
    "Sau đó là",
    "Hãy cùng xem",
    "Giờ hãy xem",
    "Tiếp theo",
    "Sau đó",
]
DANGLING_TRAILING_WORDS = {
    "bằng",
    "cho",
    "của",
    "để",
    "đến",
    "từ",
    "vào",
    "với",
}
DANGLING_TRAILING_PHRASES = {
    "để tiến hành",
    "khởi tạo cho",
    "liên quan đến",
    "phụ thuộc vào",
}


class SrtToolError(RuntimeError):
    """Expected user-facing error."""


def read_text_file(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise SrtToolError(f"Cannot decode file: {path}")


def timestamp_to_ms(value: str) -> int:
    hours = int(value[0:2])
    minutes = int(value[3:5])
    seconds = int(value[6:8])
    millis = int(value[9:12])
    return (((hours * 60 + minutes) * 60) + seconds) * 1000 + millis


def ms_to_timestamp(value: int) -> str:
    value = max(0, int(value))
    hours, remainder = divmod(value, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def timeline_duration_ms(timeline: str) -> int:
    match = TIMELINE_DETAIL_RE.match(timeline)
    if not match:
        raise SrtToolError(f"Invalid SRT timeline: {timeline!r}")
    return max(1, timestamp_to_ms(match.group("end")) - timestamp_to_ms(match.group("start")))


def max_chars_for_duration(duration_ms: int, target_cps: float, min_chars: int) -> int:
    return max(min_chars, int(round((duration_ms / 1000) * target_cps)))


def timeline_parts(timeline: str) -> tuple[int, int, str]:
    match = TIMELINE_OUTPUT_RE.match(timeline)
    if not match:
        raise SrtToolError(f"Invalid SRT timeline: {timeline!r}")
    return (
        timestamp_to_ms(match.group("start")),
        timestamp_to_ms(match.group("end")),
        match.group("rest") or "",
    )


def format_timeline(start_ms: int, end_ms: int, rest: str = "") -> str:
    return f"{ms_to_timestamp(start_ms)} --> {ms_to_timestamp(end_ms)}{rest}"


def parse_srt(path: Path) -> list[dict[str, Any]]:
    content = read_text_file(path).replace("\r\n", "\n").replace("\r", "\n")
    blocks: list[list[str]] = []
    current: list[str] = []

    for line in content.split("\n"):
        if line.strip() == "":
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)
    if current:
        blocks.append(current)

    cues: list[dict[str, Any]] = []
    for sequence, block in enumerate(blocks, start=1):
        if len(block) < 2:
            raise SrtToolError(f"Invalid SRT block at cue {sequence}: missing timeline")

        index_line = block[0].strip()
        timeline = block[1].strip()
        if not TIMELINE_RE.match(timeline):
            raise SrtToolError(
                f"Invalid SRT block at cue {sequence}: timeline not found after index {index_line!r}"
            )

        try:
            cue_id = int(index_line)
        except ValueError:
            cue_id = sequence

        cues.append(
            {
                "id": cue_id,
                "index": index_line,
                "timeline": timeline,
                "text_lines": block[2:],
            }
        )

    return cues


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SrtToolError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def output_path_for(source_path: Path, suffix: str = ".vi.srt") -> Path:
    return project_dir_for(source_path) / f"{source_path.stem}{suffix}"


def project_dir_for(source_path: Path) -> Path:
    return source_path.with_suffix("")


def work_dir_for(source_path: Path, value: str | None) -> Path:
    if value:
        return Path(value)
    return project_dir_for(source_path) / "work"


def path_from_work(work_dir: Path, filename: str) -> Path:
    return work_dir / filename


def command_extract(args: argparse.Namespace) -> None:
    source = Path(args.srt_file)
    work_dir = work_dir_for(source, args.work_dir)
    cues = parse_srt(source)
    target_cps = float(getattr(args, "target_cps", DEFAULT_TARGET_CPS))
    min_cue_chars = int(getattr(args, "min_cue_chars", DEFAULT_MIN_CUE_CHARS))

    timeline = {
        "source": str(source),
        "cue_count": len(cues),
        "target_cps": target_cps,
        "min_cue_chars": min_cue_chars,
        "cues": [
            {
                "id": cue["id"],
                "index": cue["index"],
                "timeline": cue["timeline"],
                "duration_ms": timeline_duration_ms(cue["timeline"]),
                "max_chars": max_chars_for_duration(
                    timeline_duration_ms(cue["timeline"]),
                    target_cps=target_cps,
                    min_chars=min_cue_chars,
                ),
                "text_line_count": len(cue["text_lines"]),
            }
            for cue in cues
        ],
    }
    text_rows = [
        {
            "id": cue["id"],
            "text": "\n".join(cue["text_lines"]),
            "duration_ms": timeline_duration_ms(cue["timeline"]),
            "duration_seconds": round(timeline_duration_ms(cue["timeline"]) / 1000, 3),
            "max_chars": max_chars_for_duration(
                timeline_duration_ms(cue["timeline"]),
                target_cps=target_cps,
                min_chars=min_cue_chars,
            ),
        }
        for cue in cues
    ]

    write_json(path_from_work(work_dir, "timeline.json"), timeline)
    write_jsonl(path_from_work(work_dir, "text.jsonl"), text_rows)
    print(f"Extracted {len(cues)} cues")
    print(f"Wrote {path_from_work(work_dir, 'timeline.json')}")
    print(f"Wrote {path_from_work(work_dir, 'text.jsonl')}")


def parse_model_list(value: str | None) -> list[str]:
    if not value:
        return DEFAULT_MODELS[:]
    models = [item.strip() for item in value.split(",") if item.strip()]
    if not models:
        raise SrtToolError("--models was provided but no model IDs were found")
    if len(models) > 1:
        raise SrtToolError("Only one model is supported; fallback model lists are disabled")
    return models


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def chat_completions_url(base_url: str) -> str:
    return f"{normalize_base_url(base_url)}/chat/completions"


def models_url(base_url: str) -> str:
    return f"{normalize_base_url(base_url)}/models"


def request_json(
    url: str,
    payload: dict[str, Any] | None = None,
    api_key: str | None = None,
    timeout: int = 120,
) -> Any:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-Title"] = "SRT Translate CLI"
        if "openrouter.ai" in url.lower():
            headers["HTTP-Referer"] = "https://openrouter.ai"

    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="GET" if data is None else "POST")

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise SrtToolError(f"HTTP {exc.code}: {details[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise SrtToolError(f"Network error: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SrtToolError(f"Invalid JSON response from {url}: {exc}") from exc


def http_status_from_error(error: Exception) -> int | None:
    match = re.search(r"HTTP\s+(\d{3})", str(error))
    return int(match.group(1)) if match else None


def get_available_models(base_url: str, timeout: int) -> set[str]:
    data = request_json(models_url(base_url), timeout=timeout)
    models: set[str] = set()
    for item in data.get("data", []):
        model_id = item.get("id")
        if model_id:
            models.add(model_id)
    return models


def filter_models(models: list[str], skip_check: bool, timeout: int, base_url: str) -> list[str]:
    if skip_check:
        return models
    try:
        available = get_available_models(base_url=base_url, timeout=timeout)
    except SrtToolError as exc:
        print(f"Warning: could not verify model list, using configured models: {exc}", file=sys.stderr)
        return models

    filtered = [model for model in models if model in available]
    missing = [model for model in models if model not in filtered]
    if missing:
        print(f"Skipping unavailable models: {', '.join(missing)}", file=sys.stderr)
    if not filtered:
        raise SrtToolError("No configured models are currently available")
    return filtered


def build_batches(rows: list[dict[str, Any]], batch_chars: int) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0

    for row in rows:
        row_chars = len(str(row.get("text", ""))) + 64
        if current and current_chars + row_chars > batch_chars:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(row)
        current_chars += row_chars

    if current:
        batches.append(current)
    return batches


def target_language_name(target: str) -> str:
    aliases = {
        "vi": "Vietnamese",
        "vietnamese": "Vietnamese",
        "en": "English",
    }
    return aliases.get(target.lower(), target)


def translation_style_rules(target: str) -> str:
    if target.lower() in {"vi", "vietnamese"}:
        return (
            "- Translate into natural Vietnamese subtitle style for a programming YouTube video.\n"
            "- Prioritize cue-by-cue alignment over smooth paragraph flow.\n"
            "- Keep each cue concise and easy to read aloud; avoid long academic phrasing.\n"
            "- Keep each output id aligned to the same input id. Do not move ideas, clauses, sentence endings, or connector words to another id.\n"
            "- If an input cue is a sentence fragment, translate it as a concise fragment; do not complete it with text from nearby cues.\n"
            "- If an input cue ends mid-phrase, let the Vietnamese cue also be a natural fragment or shortened equivalent.\n"
            "- If a cue is too dense for its timestamp, shorten wording inside that cue instead of borrowing text or time from adjacent cues.\n"
            "- Use common Vietnamese technical terms when they sound natural, for example "
            "'design pattern' -> 'mẫu thiết kế', 'developer' -> 'lập trình viên', "
            "'maintain' -> 'bảo trì'.\n"
            "- Keep English technical words only when Vietnamese would sound forced, for example "
            "code, framework, API, class, object, interface.\n"
            "- Lightly fix obvious speech-to-text mistakes from context, such as product names, "
            "book names, and well-known software terms.\n"
            "- Do not overuse 'bạn' if the sentence sounds more natural without it.\n"
            "- Preserve jokes and tone, but localize phrasing so it sounds like Vietnamese speech.\n"
        )
    return (
        "- Translate naturally for subtitle viewing, not word-for-word.\n"
        "- Keep each cue concise and easy to read.\n"
    )


def build_translation_messages(batch: list[dict[str, Any]], target: str) -> list[dict[str, str]]:
    target_name = target_language_name(target)
    input_payload: list[dict[str, Any]] = []
    for row in batch:
        item: dict[str, Any] = {"id": row["id"], "text": row.get("text", "")}
        if "duration_seconds" in row:
            item["duration_seconds"] = row["duration_seconds"]
        elif "duration_ms" in row:
            item["duration_seconds"] = round(float(row["duration_ms"]) / 1000, 3)
        if "max_chars" in row:
            item["max_chars"] = row["max_chars"]
        input_payload.append(item)
    system = (
        "You are a professional subtitle translator and editor. Translate subtitle text only. "
        "Return valid JSON only, with no markdown and no explanation."
    )
    user = (
        f"Translate the following English subtitle cues to {target_name}.\n"
        "Rules:\n"
        "- Return a JSON array of objects with exactly these keys: id, text.\n"
        "- Keep every id unchanged and return one object per input object.\n"
        "- Treat source cue boundaries as authoritative. Translate each object using only that object's text.\n"
        "- Do not rewrite across multiple cues to make smoother Vietnamese sentences.\n"
        "- Some input objects include duration_seconds and max_chars. Treat max_chars as a hard subtitle budget whenever possible.\n"
        "- Prefer short Vietnamese fragments that fit the timestamp over complete but late sentences.\n"
        "- Never compensate for a short cue by moving its content into a neighboring cue.\n"
        f"{translation_style_rules(target)}"
        "- Keep programming terms such as .NET, MVC, Clean Architecture, controller, view, model when natural.\n"
        "- Preserve code tokens, class names, file names, commands, URLs, and bracketed sound markers.\n"
        "- Do not add commentary.\n\n"
        f"Input JSON:\n{json.dumps(input_payload, ensure_ascii=False)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_json_array(text: str) -> list[Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("[")
        end = stripped.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, list):
        raise SrtToolError("Model response was JSON but not an array")
    return parsed


def validate_translated_batch(
    input_batch: list[dict[str, Any]], translated: list[Any]
) -> list[dict[str, Any]]:
    expected_ids = [row["id"] for row in input_batch]
    output_by_id: dict[int, str] = {}

    for item in translated:
        if not isinstance(item, dict):
            raise SrtToolError("Translated batch contains a non-object item")
        if "id" not in item or "text" not in item:
            raise SrtToolError("Translated batch item is missing id or text")
        try:
            item_id = int(item["id"])
        except (TypeError, ValueError) as exc:
            raise SrtToolError(f"Translated batch contains invalid id: {item.get('id')!r}") from exc
        output_by_id[item_id] = str(item["text"]).strip()

    missing = [cue_id for cue_id in expected_ids if cue_id not in output_by_id]
    extra = [cue_id for cue_id in output_by_id if cue_id not in set(expected_ids)]
    if missing or extra:
        raise SrtToolError(f"Translated ids mismatch. Missing={missing} Extra={extra}")

    return [{"id": cue_id, "text": output_by_id[cue_id]} for cue_id in expected_ids]


def translate_batch_once(
    batch: list[dict[str, Any]],
    model: str,
    api_key: str,
    target: str,
    timeout: int,
    base_url: str,
) -> list[dict[str, Any]]:
    payload = {
        "model": model,
        "messages": build_translation_messages(batch, target),
        "temperature": 0.1,
        "max_tokens": 4096,
    }
    data = request_json(
        chat_completions_url(base_url),
        payload=payload,
        api_key=api_key,
        timeout=timeout,
    )
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise SrtToolError(f"Unexpected chat completion response shape: {data}") from exc

    translated = extract_json_array(content)
    return validate_translated_batch(batch, translated)


def translate_batch_with_fallbacks(
    batch: list[dict[str, Any]],
    models: list[str],
    api_key: str,
    target: str,
    retries: int,
    timeout: int,
    base_url: str,
) -> tuple[list[dict[str, Any]], str]:
    last_error: Exception | None = None
    for model in models:
        for attempt in range(1, retries + 1):
            try:
                return translate_batch_once(batch, model, api_key, target, timeout, base_url), model
            except Exception as exc:
                last_error = exc
                status = http_status_from_error(exc)
                retryable = status in RETRYABLE_STATUS_CODES or status is None
                if attempt >= retries or not retryable:
                    print(f"Model {model} failed: {exc}", file=sys.stderr)
                    break
                delay = min(60, 2 ** (attempt - 1))
                print(
                    f"Model {model} attempt {attempt}/{retries} failed, retrying in {delay}s: {exc}",
                    file=sys.stderr,
                )
                time.sleep(delay)
    raise SrtToolError(f"All models failed for batch. Last error: {last_error}")


def load_existing_translations(path: Path) -> dict[int, dict[str, Any]]:
    existing: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(path):
        try:
            cue_id = int(row["id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SrtToolError(f"Invalid translated row in {path}: {row}") from exc
        existing[cue_id] = {"id": cue_id, "text": str(row.get("text", ""))}
    return existing


def command_translate(args: argparse.Namespace) -> None:
    source = Path(args.srt_file) if getattr(args, "srt_file", None) else None
    work_dir = work_dir_for(source, args.work_dir) if source else Path(args.work_dir or "work")
    text_path = path_from_work(work_dir, "text.jsonl")
    translated_path = path_from_work(work_dir, "translated.jsonl")
    cache_dir = work_dir / "cache"
    base_url = normalize_base_url(args.base_url)

    rows = read_jsonl(text_path)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SrtToolError(f"No text rows found in {text_path}. Run extract first.")

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key and args.require_api_key and not args.dry_run_copy:
        raise SrtToolError(f"{args.api_key_env} is not set")

    existing = load_existing_translations(translated_path)
    pending = [row for row in rows if int(row["id"]) not in existing]

    if args.dry_run_copy:
        copied = [{"id": int(row["id"]), "text": str(row.get("text", ""))} for row in pending]
        append_jsonl(translated_path, copied)
        print(f"Copied {len(copied)} untranslated rows to {translated_path}")
        return

    models = filter_models(
        parse_model_list(args.models),
        skip_check=args.skip_model_check,
        timeout=args.timeout,
        base_url=base_url,
    )
    batches = build_batches(pending, args.batch_chars)
    print(f"Rows total={len(rows)} existing={len(existing)} pending={len(pending)} batches={len(batches)}")
    print(f"Base URL: {base_url}")
    print(f"Models: {', '.join(models)}")

    translated_count = 0
    for batch_number, batch in enumerate(batches, start=1):
        batch_ids = [int(row["id"]) for row in batch]
        cache_path = cache_dir / f"batch-{batch_ids[0]}-{batch_ids[-1]}.json"

        if cache_path.exists():
            cached = read_json(cache_path)
            translated_rows = validate_translated_batch(batch, cached["rows"])
            model_used = cached.get("model", "cache")
        else:
            print(f"Translating batch {batch_number}/{len(batches)} ids={batch_ids[0]}..{batch_ids[-1]}")
            translated_rows, model_used = translate_batch_with_fallbacks(
                batch=batch,
                models=models,
                api_key=api_key or "",
                target=args.target,
                retries=args.retries,
                timeout=args.timeout,
                base_url=base_url,
            )
            write_json(
                cache_path,
                {
                    "model": model_used,
                    "target": args.target,
                    "ids": batch_ids,
                    "rows": translated_rows,
                },
            )

        append_jsonl(translated_path, translated_rows)
        translated_count += len(translated_rows)
        print(f"Wrote {len(translated_rows)} rows from {model_used}")

    print(f"Translation complete. Added {translated_count} rows to {translated_path}")


def wrap_subtitle_text(text: str, width: int) -> list[str]:
    if width <= 0:
        return text.splitlines() or [text]

    text = text.strip()
    if not text:
        return [""]

    wrapped_lines: list[str] = []
    for paragraph in text.splitlines() or [text]:
        wrapped = textwrap.wrap(
            paragraph,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        wrapped_lines.extend(wrapped or [""])
    return wrapped_lines


def rebalance_trailing_transitions_by_id(
    cue_ids: list[int],
    translated: dict[int, dict[str, Any]],
) -> int:
    moved_count = 0
    phrases = sorted(TRAILING_TRANSITION_PHRASES, key=len, reverse=True)

    for index, cue_id in enumerate(cue_ids[:-1]):
        next_id = cue_ids[index + 1]
        current_text = translated.get(cue_id, {"text": ""})["text"].strip()
        next_text = translated.get(next_id, {"text": ""})["text"].strip()
        if not current_text or not next_text:
            continue

        for phrase in phrases:
            pattern = re.compile(
                rf"^(?P<prefix>.+?)(?P<sep>[.!?…]\s+|\s+)({re.escape(phrase)})(?P<trail>[.!?…]*)$",
                re.IGNORECASE,
            )
            match = pattern.match(current_text)
            if not match:
                continue

            prefix = match.group("prefix").strip()
            sep = match.group("sep")
            if sep and sep[0] in ".!?…" and not prefix.endswith(tuple(".!?…")):
                prefix += sep[0]
            moved = current_text[match.start(3) : match.end(3)].strip()
            if not prefix or next_text.lower().startswith(moved.lower()):
                continue

            translated[cue_id] = {"id": cue_id, "text": prefix}
            translated[next_id] = {"id": next_id, "text": f"{moved} {next_text}".strip()}
            moved_count += 1
            break

    return moved_count


def text_ends_with_dangling_phrase(text: str) -> bool:
    normalized = re.sub(r"[.!?…\s]+$", "", text.lower()).strip()
    if not normalized:
        return False
    words = normalized.split()
    if words and words[-1] in DANGLING_TRAILING_WORDS:
        return True
    return any(normalized.endswith(phrase) for phrase in DANGLING_TRAILING_PHRASES)


def split_leading_movable_phrase(text: str, max_chars: int = 48) -> tuple[str, str] | None:
    text = text.strip()
    if not text:
        return None

    punctuation = re.search(r"[.!?…]", text)
    if punctuation:
        move = text[: punctuation.end()].strip()
        rest = text[punctuation.end() :].strip()
        if rest and len(move) <= max_chars:
            return move, rest

    words = text.split()
    if len(words) < 2:
        return None
    for word_count in range(min(4, len(words) - 1), 0, -1):
        move = " ".join(words[:word_count]).strip()
        rest = " ".join(words[word_count:]).strip()
        if move and rest and len(move) <= max_chars:
            return move, rest
    return None


def rebalance_dangling_phrases_by_id(
    cue_ids: list[int],
    translated: dict[int, dict[str, Any]],
) -> int:
    moved_count = 0
    for index, cue_id in enumerate(cue_ids[:-1]):
        next_id = cue_ids[index + 1]
        current_text = translated.get(cue_id, {"text": ""})["text"].strip()
        next_text = translated.get(next_id, {"text": ""})["text"].strip()
        if not current_text or not next_text:
            continue
        if not text_ends_with_dangling_phrase(current_text):
            continue

        split = split_leading_movable_phrase(next_text)
        if split is None:
            continue
        moved, remaining = split
        translated[cue_id] = {"id": cue_id, "text": f"{current_text} {moved}".strip()}
        translated[next_id] = {"id": next_id, "text": remaining}
        moved_count += 1
    return moved_count


def shift_timeline(timeline: str, offset_ms: int) -> str:
    start_ms, end_ms, rest = timeline_parts(timeline)
    shifted_start = max(0, start_ms + offset_ms)
    shifted_end = max(shifted_start + 1, end_ms + offset_ms)
    return format_timeline(shifted_start, shifted_end, rest)


def shift_timelines(cues: list[dict[str, Any]], offset_ms: int) -> int:
    if offset_ms == 0:
        return 0
    for cue in cues:
        cue["timeline"] = shift_timeline(str(cue["timeline"]), offset_ms)
    return len(cues)


def find_overlapping_timelines(cues: list[dict[str, Any]]) -> list[dict[str, int]]:
    overlaps: list[dict[str, int]] = []
    for index in range(len(cues) - 1):
        current = cues[index]
        next_cue = cues[index + 1]
        _, current_end, _ = timeline_parts(str(current["timeline"]))
        next_start, _, _ = timeline_parts(str(next_cue["timeline"]))
        if current_end > next_start:
            overlaps.append(
                {
                    "current_id": int(current["id"]),
                    "next_id": int(next_cue["id"]),
                    "overlap_ms": current_end - next_start,
                }
            )
    return overlaps


def remove_overlapping_timelines(
    cues: list[dict[str, Any]],
    min_duration_ms: int = 300,
    min_gap_ms: int = 1,
) -> int:
    fixed_count = 0
    for index in range(len(cues) - 1):
        current = cues[index]
        next_cue = cues[index + 1]
        current_start, current_end, current_rest = timeline_parts(str(current["timeline"]))
        next_start, _, _ = timeline_parts(str(next_cue["timeline"]))
        target_end = next_start - min_gap_ms
        if current_end <= target_end:
            continue

        if target_end <= current_start:
            new_start = max(0, target_end - min_duration_ms)
            new_end = max(new_start + 1, target_end)
            current["timeline"] = format_timeline(new_start, new_end, current_rest)
        else:
            current["timeline"] = format_timeline(current_start, target_end, current_rest)
        fixed_count += 1
    return fixed_count


def command_merge(args: argparse.Namespace) -> None:
    source = Path(args.srt_file)
    work_dir = work_dir_for(source, args.work_dir)
    timeline_path = path_from_work(work_dir, "timeline.json")
    translated_path = path_from_work(work_dir, "translated.jsonl")
    output = Path(args.output) if args.output else output_path_for(source, suffix=args.suffix)

    timeline = read_json(timeline_path)
    translated = load_existing_translations(translated_path)
    cues = timeline.get("cues", [])
    if not cues:
        raise SrtToolError(f"No cues found in {timeline_path}")

    missing = [cue["id"] for cue in cues if int(cue["id"]) not in translated]
    if missing and not args.allow_missing:
        sample = ", ".join(str(item) for item in missing[:10])
        raise SrtToolError(f"Missing translations for {len(missing)} cues. First missing ids: {sample}")

    rebalance_count = 0
    if args.rebalance_transitions:
        rebalance_count = rebalance_trailing_transitions_by_id(
            cue_ids=[int(cue["id"]) for cue in cues],
            translated=translated,
        )

    dangling_count = 0
    if args.fix_dangling:
        dangling_count = rebalance_dangling_phrases_by_id(
            cue_ids=[int(cue["id"]) for cue in cues],
            translated=translated,
        )

    overlap_count = 0
    if args.fix_overlaps:
        overlap_count = remove_overlapping_timelines(cues)

    shifted_count = shift_timelines(cues, int(args.shift_ms))
    remaining_overlaps = find_overlapping_timelines(cues)
    if remaining_overlaps and args.fix_overlaps:
        sample = ", ".join(
            f"{item['current_id']}/{item['next_id']}={item['overlap_ms']}ms"
            for item in remaining_overlaps[:5]
        )
        raise SrtToolError(f"Output still has overlapping timelines: {sample}")

    blocks: list[str] = []
    for cue in cues:
        cue_id = int(cue["id"])
        text = translated.get(cue_id, {"text": ""})["text"]
        text_lines = wrap_subtitle_text(text, width=args.wrap)
        blocks.append("\n".join([str(cue["index"]), str(cue["timeline"]), *text_lines]))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(blocks) + "\n", encoding="utf-8", newline="\n")
    print(f"Merged {len(cues)} cues into {output}")
    if args.rebalance_transitions:
        print(f"Rebalanced trailing transition phrases: {rebalance_count}")
    if args.fix_dangling:
        print(f"Fixed dangling translated phrases: {dangling_count}")
    if args.fix_overlaps:
        print(f"Fixed overlapping timelines: {overlap_count}")
    if args.shift_ms:
        print(f"Shifted timelines by {args.shift_ms} ms: {shifted_count}")
    if remaining_overlaps:
        sample = ", ".join(
            f"{item['current_id']}/{item['next_id']}={item['overlap_ms']}ms"
            for item in remaining_overlaps[:5]
        )
        print(f"Output timeline overlaps preserved from source: {len(remaining_overlaps)} ({sample})")
    else:
        print("Output timeline overlaps: 0")


def command_all(args: argparse.Namespace) -> None:
    command_extract(args)
    command_translate(args)
    command_merge(args)


def command_validate(args: argparse.Namespace) -> None:
    source = Path(args.srt_file)
    work_dir = work_dir_for(source, args.work_dir)
    source_cues = parse_srt(source)
    timeline = read_json(path_from_work(work_dir, "timeline.json"))
    translated = load_existing_translations(path_from_work(work_dir, "translated.jsonl"))

    timeline_cues = timeline.get("cues", [])
    if len(source_cues) != len(timeline_cues):
        raise SrtToolError(f"Cue count mismatch: source={len(source_cues)} timeline={len(timeline_cues)}")

    for source_cue, timeline_cue in zip(source_cues, timeline_cues):
        if source_cue["timeline"] != timeline_cue["timeline"]:
            raise SrtToolError(f"Timeline mismatch at cue {source_cue['id']}")

    missing = [cue["id"] for cue in timeline_cues if int(cue["id"]) not in translated]
    empty = [
        cue["id"]
        for cue in timeline_cues
        if int(cue["id"]) in translated and not translated[int(cue["id"])]["text"]
    ]

    print(f"Source cues: {len(source_cues)}")
    print(f"Timeline cues: {len(timeline_cues)}")
    print(f"Translated cues: {len(translated)}")
    print(f"Missing translations: {len(missing)}")
    print(f"Empty translations: {len(empty)}")
    if missing and not args.allow_missing:
        raise SrtToolError(f"Missing translations. First missing ids: {missing[:10]}")


def add_common_work_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--work-dir",
        help="Directory for timeline/text/translated files, default is <srt stem>/work",
    )


def add_timing_budget_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--target-cps",
        type=float,
        default=DEFAULT_TARGET_CPS,
        help="Target translated subtitle characters per second used during extract",
    )
    parser.add_argument(
        "--min-cue-chars",
        type=int,
        default=DEFAULT_MIN_CUE_CHARS,
        help="Minimum max_chars budget per cue used during extract",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract SRT text, translate it through OpenAI-compatible APIs, and merge it back into SRT."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="Extract timeline.json and text.jsonl from an SRT")
    extract.add_argument("srt_file")
    add_common_work_arg(extract)
    add_timing_budget_args(extract)
    extract.set_defaults(func=command_extract)

    translate = subparsers.add_parser("translate", help="Translate text.jsonl into translated.jsonl")
    add_common_work_arg(translate)
    translate.add_argument("--srt-file", help="Source SRT path used to derive default --work-dir")
    translate.add_argument("--target", default="vi", help="Target language code or name")
    translate.add_argument("--models", help=f"Single model ID, default is {DEFAULT_MODEL}")
    translate.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible API base URL")
    translate.add_argument("--api-key-env", default="NINEROUTER_API_KEY", help="Environment variable containing API key")
    translate.add_argument("--require-api-key", action="store_true", help="Fail if --api-key-env is not set")
    translate.add_argument("--batch-chars", type=int, default=6000, help="Approximate max chars per batch")
    translate.add_argument("--limit", type=int, help="Translate only the first N extracted rows")
    translate.add_argument("--retries", type=int, default=3, help="Retries per model per batch")
    translate.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds")
    translate.add_argument("--skip-model-check", action="store_true", help="Do not verify models against /api/v1/models")
    translate.add_argument(
        "--dry-run-copy",
        action="store_true",
        help="Copy source text into translated.jsonl without calling the translation API",
    )
    translate.set_defaults(func=command_translate)

    merge = subparsers.add_parser("merge", help="Merge translated.jsonl back into an SRT")
    merge.add_argument("srt_file")
    add_common_work_arg(merge)
    merge.add_argument("--output", help="Output SRT path")
    merge.add_argument("--suffix", default=".vi.srt", help="Output suffix when --output is omitted")
    merge.add_argument("--wrap", type=int, default=0, help="Wrap subtitle text to this width; 0 disables wrapping")
    merge.add_argument(
        "--shift-ms",
        type=int,
        default=0,
        help="Shift all output cue timestamps by this many milliseconds; negative makes subtitles earlier",
    )
    merge.add_argument(
        "--rebalance-transitions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Move short trailing transition phrases to the next cue without changing timestamps",
    )
    merge.add_argument(
        "--fix-overlaps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shorten earlier cues when adjacent SRT timelines overlap",
    )
    merge.add_argument(
        "--fix-dangling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Move a short leading phrase from the next cue when a translated cue ends mid-phrase; off by default to preserve source cue alignment",
    )
    merge.add_argument("--allow-missing", action="store_true", help="Allow missing translated cues as blank text")
    merge.set_defaults(func=command_merge)

    all_cmd = subparsers.add_parser("all", help="Run extract, translate, and merge")
    all_cmd.add_argument("srt_file")
    add_common_work_arg(all_cmd)
    add_timing_budget_args(all_cmd)
    all_cmd.add_argument("--target", default="vi", help="Target language code or name")
    all_cmd.add_argument("--models", help=f"Single model ID, default is {DEFAULT_MODEL}")
    all_cmd.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible API base URL")
    all_cmd.add_argument("--api-key-env", default="NINEROUTER_API_KEY", help="Environment variable containing API key")
    all_cmd.add_argument("--require-api-key", action="store_true", help="Fail if --api-key-env is not set")
    all_cmd.add_argument("--batch-chars", type=int, default=6000, help="Approximate max chars per batch")
    all_cmd.add_argument("--limit", type=int, help="Translate only the first N extracted rows")
    all_cmd.add_argument("--retries", type=int, default=3, help="Retries per model per batch")
    all_cmd.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds")
    all_cmd.add_argument("--skip-model-check", action="store_true", help="Do not verify models against /api/v1/models")
    all_cmd.add_argument("--dry-run-copy", action="store_true", help="Copy source text without calling the translation API")
    all_cmd.add_argument("--output", help="Output SRT path")
    all_cmd.add_argument("--suffix", default=".vi.srt", help="Output suffix when --output is omitted")
    all_cmd.add_argument("--wrap", type=int, default=0, help="Wrap subtitle text to this width; 0 disables wrapping")
    all_cmd.add_argument(
        "--shift-ms",
        type=int,
        default=0,
        help="Shift all output cue timestamps by this many milliseconds; negative makes subtitles earlier",
    )
    all_cmd.add_argument(
        "--rebalance-transitions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Move short trailing transition phrases to the next cue without changing timestamps",
    )
    all_cmd.add_argument(
        "--fix-overlaps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shorten earlier cues when adjacent SRT timelines overlap",
    )
    all_cmd.add_argument(
        "--fix-dangling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Move a short leading phrase from the next cue when a translated cue ends mid-phrase; off by default to preserve source cue alignment",
    )
    all_cmd.add_argument("--allow-missing", action="store_true", help="Allow missing translated cues as blank text")
    all_cmd.set_defaults(func=command_all)

    validate = subparsers.add_parser("validate", help="Validate extracted timeline and translated rows")
    validate.add_argument("srt_file")
    add_common_work_arg(validate)
    validate.add_argument("--allow-missing", action="store_true", help="Do not fail on missing translations")
    validate.set_defaults(func=command_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except SrtToolError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

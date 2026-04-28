#!/usr/bin/env python3
"""Create timeline-aligned Vietnamese voice-over audio from an SRT file.

Each subtitle cue is synthesized separately with Edge TTS, fitted into its
subtitle time slot, then assembled into one audio file with silence gaps.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import edge_tts
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


TIMELINE_RE = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+"
    r"(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
)
BRACKET_ONLY_RE = re.compile(r"^\s*(\[[^\]]+\]|\([^)]+\))\s*$")
PUNCTUATION_ONLY_RE = re.compile(r"^[\s,.;:!?…-]+$")
ENGLISH_TOKEN_RE = re.compile(
    r"(?<![\w.+#-])(?:"
    r"\.[A-Za-z][A-Za-z0-9]*(?:\.[A-Za-z0-9]+)*"
    r"|[A-Za-z][A-Za-z0-9]*(?:[.+#_-][A-Za-z0-9]+)+"
    r"|[A-Z]{2,}[A-Za-z0-9]*"
    r"|[A-Za-z]+"
    r")(?![\w.+#-])"
)
VI_ASCII_STOPWORDS = {
    "anh",
    "ban",
    "bo",
    "cach",
    "cac",
    "cho",
    "co",
    "cua",
    "da",
    "de",
    "di",
    "duoc",
    "gia",
    "hay",
    "hon",
    "khi",
    "khong",
    "kinh",
    "la",
    "lai",
    "lam",
    "mot",
    "nay",
    "nghe",
    "nguoi",
    "nhieu",
    "nhu",
    "nhung",
    "qua",
    "ra",
    "rat",
    "roi",
    "sau",
    "se",
    "ta",
    "thay",
    "thi",
    "trong",
    "tu",
    "va",
    "ve",
    "voi",
}
DEFAULT_EDGE_VOICE = "vi-VN-NamMinhNeural"
DEFAULT_PHONETIC_MAP_MODEL = "cx/gpt-5.5"
DEFAULT_PHONETIC_MAP_BASE_URL = "http://localhost:20128/v1"
DEFAULT_ENGLISH_VOICE = "en-US-GuyNeural"
SUPPORTED_ASSEMBLE_AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg")
ENGLISH_TTS_TERMS = [
    "Object.getPrototypeOf",
    "Refactoring.Guru",
    "Stack Overflow",
    "principal engineer",
    "senior developer",
    "junior developer",
    "Design Patterns",
    "Gang of Four",
    "Prototype chain",
    "static getInstance",
    "static instance",
    "side effect",
    "code base",
    "bad practice",
    "for...of",
    "TypeScript",
    "JavaScript",
    "Play-Doh",
    "Singleton",
    "Prototype",
    "Builder",
    "Factory",
    "Facade",
    "Observer",
    "Settings",
    "Vue.js",
    "jQuery",
    "C++",
    "C#",
    ".NET",
    "API",
    "UI",
    "iOS",
    "Android",
    "Internet",
    "Ukraine",
    "developer",
    "website",
    "video",
    "app",
    "codebase",
    "boilerplate",
    "interface",
    "constructor",
    "method",
    "function",
    "object",
    "class",
    "static",
    "instance",
    "new",
    "global",
    "clone",
    "subclass",
    "inheritance",
    "prototype",
    "zombie",
    "name",
    "eatBrains",
    "this",
    "render",
    "package",
    "proxy",
    "reactivity",
    "data",
    "Handler",
    "Reflect",
    "iterator",
    "collection",
    "abstraction",
    "for",
    "range",
    "next",
    "done",
    "pull",
    "push",
    "code",
]
PHONETIC_ENGLISH_MAP = {
    "Object.getPrototypeOf": "ốp dếch chấm ghét prô tô taip ốp",
    "Refactoring.Guru": "ri phác tơ ring gu ru",
    "Stack Overflow": "sờ tác ô vờ phờ lâu",
    "principal engineer": "prin xi pồ en gi nia",
    "senior developer": "xi nia đì veo lơ pờ",
    "junior developer": "chu nia đì veo lơ pờ",
    "Design Patterns": "đì dai pát tờn",
    "Gang of Four": "geng ốp pho",
    "Prototype chain": "prô tô taip chên",
    "static getInstance": "sờ ta tích ghét in sờ tần",
    "static instance": "sờ ta tích in sờ tần",
    "side effect": "sai đì phéc",
    "code base": "cốt bêis",
    "bad practice": "bét prac tít",
    "for...of": "pho ốp",
    "TypeScript": "taip sờ cript",
    "JavaScript": "gia va sờ cript",
    "Play-Doh": "pờ lây đô",
    "Singleton": "sing gồ tần",
    "Prototype": "prô tô taip",
    "Builder": "bin đờ",
    "Factory": "phác tơ ri",
    "Facade": "pha xát",
    "Observer": "ốp dơ vờ",
    "Settings": "sét ting",
    "Vue.js": "viu chấm j s",
    "jQuery": "j quê ri",
    "C++": "xi cộng cộng",
    "C#": "xi sáp",
    ".NET": "đót nét",
    "API": "ây pi ai",
    "UI": "iu ai",
    "iOS": "ai ô ét",
    "Android": "an đroi",
    "Internet": "in tơ nét",
    "Ukraine": "iu krên",
    "developer": "đì veo lơ pờ",
    "website": "quép sai",
    "video": "vi đi ô",
    "app": "áp",
    "codebase": "cốt bêis",
    "boilerplate": "boi lờ plêt",
    "interface": "in tờ phêis",
    "constructor": "con sờ trắc tờ",
    "method": "mé thợt",
    "function": "phăng sần",
    "object": "ốp dếch",
    "class": "cờ lát",
    "static": "sờ ta tích",
    "instance": "in sờ tần",
    "new": "niu",
    "global": "gờ lâu bồ",
    "clone": "cờ lôn",
    "subclass": "sắp cờ lát",
    "inheritance": "in he ri tần",
    "prototype": "prô tô taip",
    "zombie": "dom bi",
    "name": "nêm",
    "eatBrains": "ít brên",
    "this": "đít",
    "render": "ren đờ",
    "package": "pác kịch",
    "proxy": "próc xi",
    "reactivity": "ri ác ti vi ti",
    "data": "đây ta",
    "Handler": "hen đờ lờ",
    "Reflect": "ri phlect",
    "iterator": "i tơ rê tờ",
    "collection": "cơ lec sần",
    "abstraction": "áp sờ trac sần",
    "for": "pho",
    "range": "rênj",
    "next": "néxt",
    "done": "đân",
    "pull": "pun",
    "push": "pút",
    "code": "cốt",
}


class VoiceToolError(RuntimeError):
    """Expected user-facing error."""


def default_voice_work_dir(srt_path: Path) -> Path:
    return srt_path.parent / "voice_work"


def default_voice_output_path(srt_path: Path) -> Path:
    return srt_path.with_suffix(".voice.wav")


def default_assembled_output_path(srt_path: Path) -> Path:
    return srt_path.with_suffix(".assembled.wav")


def default_txt_output_dir(srt_path: Path) -> Path:
    return srt_path.with_name(f"{srt_path.stem}.txt-batch")


def audio_export_format(output_path: Path) -> str:
    export_format = output_path.suffix.lstrip(".").lower() or "wav"
    if export_format == "m4a":
        return "mp4"
    return export_format


def read_text_file(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise VoiceToolError(f"Cannot decode file: {path}")


def timestamp_to_ms(value: str) -> int:
    hours = int(value[0:2])
    minutes = int(value[3:5])
    seconds = int(value[6:8])
    millis = int(value[9:12])
    return (((hours * 60 + minutes) * 60) + seconds) * 1000 + millis


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
            continue
        match = TIMELINE_RE.match(block[1].strip())
        if not match:
            raise VoiceToolError(f"Invalid timeline at cue {sequence}: {block[1]!r}")
        text = " ".join(line.strip() for line in block[2:] if line.strip()).strip()
        cues.append(
            {
                "id": int(block[0].strip()) if block[0].strip().isdigit() else sequence,
                "start_ms": timestamp_to_ms(match.group("start")),
                "end_ms": timestamp_to_ms(match.group("end")),
                "text": text,
            }
        )
    return cues


def run_command(command: list[str]) -> None:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise VoiceToolError(
            f"Command failed: {' '.join(command)}\n{result.stderr.strip()}"
        )


def ensure_ffmpeg_available() -> None:
    for binary in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run(
                [binary, "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise VoiceToolError(f"{binary} is required but was not found in PATH") from exc


def media_duration_ms(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        raise VoiceToolError(f"Audio file is missing or empty: {path}")
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise VoiceToolError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    return int(round(float(result.stdout.strip()) * 1000))


def cache_key(text: str, voice: str, rate: str, pitch: str, volume: str) -> str:
    raw = json.dumps(
        {"text": text, "voice": voice, "rate": rate, "pitch": pitch, "volume": volume},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


async def synthesize_edge_tts(
    text: str,
    voice: str,
    output_path: Path,
    rate: str,
    pitch: str,
    volume: str,
) -> None:
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        pitch=pitch,
        volume=volume,
    )
    await communicate.save(str(output_path))


def synthesize_edge_tts_with_retries(
    text: str,
    voice: str,
    output_path: Path,
    rate: str,
    pitch: str,
    volume: str,
    retries: int,
) -> None:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        if output_path.exists():
            output_path.unlink()
        try:
            asyncio.run(
                synthesize_edge_tts(
                    text=text,
                    voice=voice,
                    output_path=output_path,
                    rate=rate,
                    pitch=pitch,
                    volume=volume,
                )
            )
            if output_path.exists() and output_path.stat().st_size > 0:
                return
            raise VoiceToolError("Edge TTS created an empty audio file")
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            delay = min(10, attempt * 2)
            print(f"Edge TTS failed, retrying in {delay}s: {exc}", file=sys.stderr)
            time.sleep(delay)
    raise VoiceToolError(f"Edge TTS failed after {retries} attempts: {last_error}")


def english_term_pattern() -> re.Pattern[str]:
    escaped = sorted((re.escape(term) for term in ENGLISH_TTS_TERMS), key=len, reverse=True)
    return re.compile(r"(?<!\w)(" + "|".join(escaped) + r")(?!\w)", re.IGNORECASE)


ENGLISH_TERM_RE = english_term_pattern()


def phonetic_english_pattern() -> re.Pattern[str]:
    escaped = sorted((re.escape(term) for term in PHONETIC_ENGLISH_MAP), key=len, reverse=True)
    return re.compile(r"(?<!\w)(" + "|".join(escaped) + r")(?!\w)", re.IGNORECASE)


PHONETIC_ENGLISH_RE = phonetic_english_pattern()


def refresh_phonetic_english_pattern() -> None:
    global PHONETIC_ENGLISH_RE
    PHONETIC_ENGLISH_RE = phonetic_english_pattern()


def load_phonetic_english_map(path: Path) -> int:
    if not path.exists():
        raise VoiceToolError(f"Phonetic map file not found: {path}")

    content = path.read_text(encoding="utf-8")
    match = re.search(
        r"(?:PHONETIC_ENGLISH_MAP_UPDATE|PHONETIC_ENGLISH_MAP)\s*=\s*(\{.*?\})\s*$",
        content,
        re.S,
    )
    if not match:
        raise VoiceToolError(
            f"Cannot find PHONETIC_ENGLISH_MAP_UPDATE or PHONETIC_ENGLISH_MAP in {path}"
        )
    try:
        loaded = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError) as exc:
        raise VoiceToolError(f"Invalid phonetic map file {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise VoiceToolError(f"Phonetic map in {path} is not a dict")

    added = 0
    for key, value in loaded.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        value = value.strip()
        if not value:
            continue
        PHONETIC_ENGLISH_MAP[key] = value
        added += 1
    refresh_phonetic_english_pattern()
    return added


def apply_phonetic_english(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        source = match.group(0)
        for key, value in PHONETIC_ENGLISH_MAP.items():
            if key.lower() == source.lower():
                return value
        return source

    return PHONETIC_ENGLISH_RE.sub(replace, text)


def canonical_english_term_map() -> dict[str, str]:
    terms: dict[str, str] = {}
    for term in [*ENGLISH_TTS_TERMS, *PHONETIC_ENGLISH_MAP]:
        terms.setdefault(term.lower(), term)
    return terms


def should_collect_english_token(
    token: str,
    known_terms: dict[str, str],
    include_lowercase: bool,
) -> bool:
    token_lower = token.lower()
    if len(token) <= 1:
        return False
    if token_lower in VI_ASCII_STOPWORDS:
        return False
    if token_lower in known_terms:
        return True
    if re.search(r"[0-9.+#_-]", token):
        return True
    if re.fullmatch(r"[A-Z]{2,}[A-Za-z0-9]*", token):
        return True
    if re.search(r"[a-z][A-Z]", token):
        return True
    return include_lowercase and bool(re.fullmatch(r"[A-Za-z]{3,}", token))


def scan_english_terms(
    cues: list[dict[str, Any]],
    include_lowercase: bool,
) -> dict[str, dict[str, Any]]:
    known_terms = canonical_english_term_map()
    found: dict[str, dict[str, Any]] = {}

    def add_term(term: str, cue_id: int) -> None:
        canonical = known_terms.get(term.lower(), term)
        row = found.setdefault(
            canonical,
            {
                "term": canonical,
                "phonetic": PHONETIC_ENGLISH_MAP.get(canonical, ""),
                "count": 0,
                "cue_ids": [],
            },
        )
        row["count"] += 1
        if cue_id not in row["cue_ids"]:
            row["cue_ids"].append(cue_id)

    for cue in cues:
        cue_id = int(cue["id"])
        text = str(cue.get("text", ""))
        occupied: list[tuple[int, int]] = []

        for match in ENGLISH_TERM_RE.finditer(text):
            add_term(match.group(0), cue_id)
            occupied.append((match.start(), match.end()))

        for match in ENGLISH_TOKEN_RE.finditer(text):
            if any(start <= match.start() < end for start, end in occupied):
                continue
            token = match.group(0)
            if should_collect_english_token(token, known_terms, include_lowercase):
                add_term(token, cue_id)

    return found


def write_english_map_template(
    path: Path,
    terms: dict[str, dict[str, Any]],
    missing_only: bool,
) -> None:
    rows = sorted(
        (
            row
            for row in terms.values()
            if not missing_only or not row["phonetic"]
        ),
        key=lambda row: row["term"].lower(),
    )
    lines = [
        "# Generated by srt_to_voice.py --scan-english-map.",
        "# Use this file with --phonetic-map-file, or leave it next to the SRT for auto-loading.",
        "PHONETIC_ENGLISH_MAP_UPDATE = {",
    ]
    for row in rows:
        term = json.dumps(row["term"], ensure_ascii=False)
        phonetic = json.dumps(row["phonetic"], ensure_ascii=False)
        cue_sample = ",".join(str(item) for item in row["cue_ids"][:8])
        lines.append(
            f"    {term}: {phonetic},  # count={row['count']} cues={cue_sample}"
        )
    lines.append("}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def request_json(
    url: str,
    payload: dict[str, Any],
    api_key: str = "",
    timeout: int = 120,
) -> Any:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        if "openrouter.ai" in url.lower():
            headers["HTTP-Referer"] = "https://openrouter.ai"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise VoiceToolError(f"HTTP {exc.code}: {details[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise VoiceToolError(f"Network error: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise VoiceToolError(f"Invalid JSON response from {url}: {exc}") from exc


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise VoiceToolError("Model response was JSON but not an object")
    return parsed


def auto_fill_english_map_template(
    path: Path,
    model: str,
    base_url: str,
    api_key: str,
    timeout: int,
) -> tuple[int, int]:
    content = path.read_text(encoding="utf-8")
    match = re.search(r"PHONETIC_ENGLISH_MAP_UPDATE\s*=\s*(\{.*?\})\s*$", content, re.S)
    if not match:
        raise VoiceToolError(f"Cannot find PHONETIC_ENGLISH_MAP_UPDATE in {path}")
    current = ast.literal_eval(match.group(1))
    if not isinstance(current, dict):
        raise VoiceToolError(f"PHONETIC_ENGLISH_MAP_UPDATE in {path} is not a dict")

    terms = list(current)
    prompt = (
        "You are preparing a Vietnamese phonetic map for text-to-speech.\n"
        "For each term, return a JSON object mapping the exact term to a Vietnamese phonetic spelling.\n"
        "Rules:\n"
        "- Use Vietnamese syllables that make an Edge Vietnamese voice pronounce the English/code term recognizably.\n"
        "- Keep Vietnamese false positives empty string, especially Vietnamese words without accents.\n"
        "- For CamelCase/code identifiers, spell them as Vietnamese-readable chunks.\n"
        "- Preserve exact keys and return JSON object only.\n\n"
        f"Terms JSON:\n{json.dumps(terms, ensure_ascii=False)}"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
    }
    data = request_json(
        f"{base_url.rstrip('/')}/chat/completions",
        payload=payload,
        api_key=api_key,
        timeout=timeout,
    )
    try:
        filled = extract_json_object(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise VoiceToolError(f"Unexpected chat completion response shape: {data}") from exc

    missing = [term for term in terms if term not in filled]
    if missing:
        raise VoiceToolError(f"Model response missing terms: {missing[:10]}")

    merged: dict[str, str] = {}
    for term in terms:
        old = str(current.get(term, "")).strip()
        new = str(filled.get(term, "")).strip()
        merged[term] = old or new

    comment_by_term: dict[str, str] = {}
    for line in content.splitlines():
        line_match = re.match(r'\s*"(?P<term>.+?)"\s*:\s*.*?(?P<comment>\s*#.*)$', line)
        if line_match:
            comment_by_term[line_match.group("term")] = line_match.group("comment")

    lines = [
        "# Generated by srt_to_voice.py --scan-english-map.",
        f"# Auto-filled with {model}. Empty values are likely Vietnamese false positives.",
        "PHONETIC_ENGLISH_MAP_UPDATE = {",
    ]
    for term in terms:
        comment = comment_by_term.get(term, "")
        lines.append(
            f"    {json.dumps(term, ensure_ascii=False)}: "
            f"{json.dumps(merged[term], ensure_ascii=False)}, {comment}".rstrip()
        )
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    filled_count = sum(1 for value in merged.values() if value)
    return filled_count, len(merged) - filled_count


def split_mixed_language_text(text: str) -> list[dict[str, str]]:
    parts: list[dict[str, str]] = []
    cursor = 0
    for match in ENGLISH_TERM_RE.finditer(text):
        if match.start() > cursor:
            parts.append({"lang": "vi", "text": text[cursor : match.start()]})
        parts.append({"lang": "en", "text": match.group(0)})
        cursor = match.end()
    if cursor < len(text):
        parts.append({"lang": "vi", "text": text[cursor:]})

    compact: list[dict[str, str]] = []
    for part in parts:
        if not part["text"]:
            continue
        if compact and compact[-1]["lang"] == part["lang"]:
            compact[-1]["text"] += part["text"]
        else:
            compact.append(part)
    return compact or [{"lang": "vi", "text": text}]


def synthesize_edge_tts_mixed_with_retries(
    text: str,
    voice: str,
    english_voice: str,
    output_path: Path,
    rate: str,
    pitch: str,
    volume: str,
    retries: int,
) -> None:
    parts = split_mixed_language_text(text)
    if not any(part["lang"] == "en" for part in parts):
        synthesize_edge_tts_with_retries(
            text=text,
            voice=voice,
            output_path=output_path,
            rate=rate,
            pitch=pitch,
            volume=volume,
            retries=retries,
        )
        return

    temp_dir = output_path.parent / f"{output_path.stem}.parts"
    temp_dir.mkdir(parents=True, exist_ok=True)
    segments: list[AudioSegment] = []
    try:
        for index, part in enumerate(parts, start=1):
            part_text = part["text"].strip()
            if not part_text:
                continue
            if PUNCTUATION_ONLY_RE.match(part_text):
                segments.append(AudioSegment.silent(duration=80))
                continue
            part_text = part_text.strip(" \t\r\n,.;:!?…")
            if not part_text:
                segments.append(AudioSegment.silent(duration=80))
                continue
            part_voice = english_voice if part["lang"] == "en" else voice
            part_path = temp_dir / f"{index:03d}-{part['lang']}.mp3"
            synthesize_edge_tts_with_retries(
                text=part_text,
                voice=part_voice,
                output_path=part_path,
                rate=rate,
                pitch=pitch,
                volume=volume,
                retries=retries,
            )
            segments.append(AudioSegment.from_file(part_path))

        if not segments:
            raise VoiceToolError("No mixed-language audio segments were created")
        combined = AudioSegment.silent(duration=0)
        for segment in segments:
            combined += segment
        combined.export(output_path, format="mp3")
    except Exception as exc:
        print(f"Mixed English TTS failed, falling back to Vietnamese-only voice: {exc}", file=sys.stderr)
        synthesize_edge_tts_with_retries(
            text=text,
            voice=voice,
            output_path=output_path,
            rate=rate,
            pitch=pitch,
            volume=volume,
            retries=retries,
        )
    finally:
        for child in temp_dir.glob("*"):
            child.unlink(missing_ok=True)
        try:
            temp_dir.rmdir()
        except OSError:
            pass


def speed_filter(speed: float) -> str:
    # atempo is reliable in the 0.5..2.0 range; this tool only speeds up.
    if speed < 0.5 or speed > 2.0:
        raise VoiceToolError(f"Unsupported speed factor for ffmpeg atempo: {speed}")
    return f"atempo={speed:.5f}"


def trim_tts_silence(
    raw_path: Path,
    prepared_path: Path,
    silence_thresh: int,
    min_silence_len: int,
    keep_silence_ms: int,
    sample_rate: int,
) -> dict[str, Any]:
    audio = AudioSegment.from_file(raw_path)
    original_ms = len(audio)
    ranges = detect_nonsilent(
        audio,
        min_silence_len=max(1, min_silence_len),
        silence_thresh=silence_thresh,
    )

    if not ranges:
        prepared = audio.set_frame_rate(sample_rate).set_channels(2)
        prepared.export(prepared_path, format="wav")
        return {
            "tts_silence_trimmed": False,
            "tts_original_ms": original_ms,
            "tts_prepared_ms": len(prepared),
            "tts_trim_start_ms": 0,
            "tts_trim_end_ms": 0,
        }

    start_ms = max(0, ranges[0][0] - keep_silence_ms)
    end_ms = min(original_ms, ranges[-1][1] + keep_silence_ms)
    prepared = audio[start_ms:end_ms].set_frame_rate(sample_rate).set_channels(2)
    prepared.export(prepared_path, format="wav")

    trim_start_ms = start_ms
    trim_end_ms = max(0, original_ms - end_ms)
    return {
        "tts_silence_trimmed": trim_start_ms > 0 or trim_end_ms > 0,
        "tts_original_ms": original_ms,
        "tts_prepared_ms": len(prepared),
        "tts_trim_start_ms": trim_start_ms,
        "tts_trim_end_ms": trim_end_ms,
    }


def fit_audio_to_slot(
    raw_path: Path,
    processed_path: Path,
    slot_ms: int,
    max_speed: float,
    fit_mode: str,
    sample_rate: int,
) -> dict[str, Any]:
    raw_ms = media_duration_ms(raw_path)
    required_speed = raw_ms / slot_ms if slot_ms > 0 else 1.0
    applied_speed = 1.0
    overflow_before_trim_ms = 0

    if required_speed > 1.0:
        applied_speed = min(required_speed, max_speed)

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(raw_path),
    ]
    if applied_speed > 1.001:
        command.extend(["-filter:a", speed_filter(applied_speed)])
    command.extend(["-ar", str(sample_rate), "-ac", "2", str(processed_path)])
    run_command(command)

    fitted_ms = media_duration_ms(processed_path)
    if fitted_ms > slot_ms:
        overflow_before_trim_ms = fitted_ms - slot_ms
        if fit_mode == "trim":
            trimmed_path = processed_path.with_suffix(".trim.wav")
            run_command(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(processed_path),
                    "-t",
                    f"{slot_ms / 1000:.3f}",
                    "-ar",
                    str(sample_rate),
                    "-ac",
                    "2",
                    str(trimmed_path),
                ]
            )
            trimmed_path.replace(processed_path)
            fitted_ms = media_duration_ms(processed_path)

    return {
        "raw_ms": raw_ms,
        "slot_ms": slot_ms,
        "required_speed": round(required_speed, 4),
        "applied_speed": round(applied_speed, 4),
        "final_ms": fitted_ms,
        "overflow_before_trim_ms": overflow_before_trim_ms,
        "trimmed": fit_mode == "trim" and overflow_before_trim_ms > 0,
    }


def should_skip_text(text: str, skip_bracket_only: bool) -> bool:
    if not text.strip():
        return True
    return bool(skip_bracket_only and BRACKET_ONLY_RE.match(text))


def load_audio_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        raise VoiceToolError(f"Manifest not found: {manifest_path}")
    try:
        manifest = json.loads(read_text_file(manifest_path))
    except json.JSONDecodeError as exc:
        raise VoiceToolError(f"Invalid manifest JSON: {manifest_path}: {exc}") from exc
    rows = manifest.get("rows") if isinstance(manifest, dict) else None
    if not isinstance(rows, list):
        raise VoiceToolError(f"Manifest must contain a rows array: {manifest_path}")
    normalized_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise VoiceToolError(f"Invalid manifest row at index {index}: expected object")
        normalized_rows.append(row)
    return normalized_rows


def audio_prefix_from_manifest_row(row: dict[str, Any]) -> str:
    txt_file = str(row.get("file") or "").strip()
    if txt_file:
        return Path(txt_file).stem
    try:
        sequence = int(row["sequence"])
        cue_id = int(row["cue_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VoiceToolError(f"Manifest row is missing file/sequence/cue_id: {row}") from exc
    return f"{sequence:05d}-cue-{cue_id:05d}"


def find_audio_for_manifest_row(audio_input_dir: Path, row: dict[str, Any]) -> Path | None:
    prefix = audio_prefix_from_manifest_row(row)
    candidates: list[Path] = []
    for path in audio_input_dir.iterdir():
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_ASSEMBLE_AUDIO_EXTENSIONS:
            continue
        if path.stem.startswith(prefix):
            candidates.append(path)

    if not candidates:
        return None

    def sort_key(path: Path) -> tuple[int, int, str]:
        suffix = path.suffix.lower()
        extension_rank = SUPPORTED_ASSEMBLE_AUDIO_EXTENSIONS.index(suffix)
        return (0 if path.stem == prefix else 1, extension_rank, path.name.lower())

    return sorted(candidates, key=sort_key)[0]


def manifest_row_int(row: dict[str, Any], key: str, row_index: int) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise VoiceToolError(f"Invalid or missing {key!r} in manifest row {row_index}") from exc


def build_voice_units(
    cues: list[dict[str, Any]],
    group_cues: int,
    max_group_gap_ms: int,
    skip_bracket_only: bool,
) -> list[dict[str, Any]]:
    group_cues = max(1, group_cues)
    units: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def flush_current() -> None:
        nonlocal current
        if current is not None:
            current["text"] = " ".join(current.pop("text_parts")).strip()
            current["id"] = (
                str(current["ids"][0])
                if len(current["ids"]) == 1
                else f"{current['ids'][0]}-{current['ids'][-1]}"
            )
            current["file_id"] = (
                f"{int(current['ids'][0]):05d}"
                if len(current["ids"]) == 1
                else f"{int(current['ids'][0]):05d}-{int(current['ids'][-1]):05d}"
            )
            units.append(current)
            current = None

    for cue in cues:
        cue_part = {
            "id": cue["id"],
            "start_ms": int(cue["start_ms"]),
            "end_ms": int(cue["end_ms"]),
            "text": str(cue["text"]).strip(),
        }
        cue_text = str(cue["text"]).strip()
        cue_is_skip = should_skip_text(cue_text, skip_bracket_only)

        if cue_is_skip:
            flush_current()
            units.append(
                {
                    "id": str(cue["id"]),
                    "file_id": f"{int(cue['id']):05d}",
                    "ids": [cue["id"]],
                    "start_ms": int(cue["start_ms"]),
                    "end_ms": int(cue["end_ms"]),
                    "text": cue_text,
                    "cue_parts": [cue_part],
                    "skipped": True,
                }
            )
            continue

        if current is None:
            current = {
                "ids": [cue["id"]],
                "start_ms": int(cue["start_ms"]),
                "end_ms": int(cue["end_ms"]),
                "text_parts": [cue_text],
                "cue_parts": [cue_part],
                "skipped": False,
            }
            continue

        gap_ms = int(cue["start_ms"]) - int(current["end_ms"])
        can_group = len(current["ids"]) < group_cues and gap_ms <= max_group_gap_ms
        if can_group:
            current["ids"].append(cue["id"])
            current["end_ms"] = int(cue["end_ms"])
            current["text_parts"].append(cue_text)
            current["cue_parts"].append(cue_part)
        else:
            flush_current()
            current = {
                "ids": [cue["id"]],
                "start_ms": int(cue["start_ms"]),
                "end_ms": int(cue["end_ms"]),
                "text_parts": [cue_text],
                "cue_parts": [cue_part],
                "skipped": False,
            }

    flush_current()
    return units


def single_cue_unit(cue_part: dict[str, Any]) -> dict[str, Any]:
    cue_id = cue_part["id"]
    return {
        "id": str(cue_id),
        "file_id": f"{int(cue_id):05d}",
        "ids": [cue_id],
        "start_ms": int(cue_part["start_ms"]),
        "end_ms": int(cue_part["end_ms"]),
        "text": str(cue_part["text"]).strip(),
        "cue_parts": [cue_part],
        "skipped": False,
    }


def command_scan_english_map(args: argparse.Namespace) -> None:
    srt_path = Path(args.srt_file)
    if args.phonetic_map_file:
        loaded_count = load_phonetic_english_map(Path(args.phonetic_map_file))
        print(f"Loaded phonetic map entries: {loaded_count}")

    cues = parse_srt(srt_path)
    if args.limit:
        cues = cues[: args.limit]
    if not cues:
        raise VoiceToolError(f"No cues found in {srt_path}")

    terms = scan_english_terms(
        cues,
        include_lowercase=args.scan_lowercase_english,
    )
    output_path = (
        Path(args.english_map_output)
        if args.english_map_output
        else srt_path.with_name(f"{srt_path.stem}.english-map.py")
    )
    write_english_map_template(
        path=output_path,
        terms=terms,
        missing_only=args.english_map_missing_only,
    )

    mapped_count = sum(1 for row in terms.values() if row["phonetic"])
    missing_count = len(terms) - mapped_count
    if args.auto_fill_english_map:
        api_key = os.environ.get(args.phonetic_map_api_key_env, "")
        filled_count, empty_count = auto_fill_english_map_template(
            path=output_path,
            model=args.phonetic_map_model,
            base_url=args.phonetic_map_base_url,
            api_key=api_key,
            timeout=args.phonetic_map_timeout,
        )
        print(f"Auto-filled phonetic map with {args.phonetic_map_model}")
        print(f"Filled phonetic entries: {filled_count}")
        print(f"Empty/false-positive entries: {empty_count}")
    print(f"Cues scanned: {len(cues)}")
    print(f"English terms found: {len(terms)}")
    print(f"Already mapped: {mapped_count}")
    print(f"Missing phonetic map: {missing_count}")
    print(f"Wrote: {output_path}")


def command_export_txt(args: argparse.Namespace) -> None:
    srt_path = Path(args.srt_file)
    cues = parse_srt(srt_path)
    if args.limit:
        cues = cues[: args.limit]
    if not cues:
        raise VoiceToolError(f"No cues found in {srt_path}")

    output_dir = Path(args.txt_output_dir) if args.txt_output_dir else default_txt_output_dir(srt_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    rows: list[dict[str, Any]] = []
    written_count = 0
    skipped_count = 0

    for sequence, cue in enumerate(cues, start=1):
        cue_id = int(cue["id"])
        text = str(cue.get("text", "")).strip()
        skipped = should_skip_text(text, args.skip_bracket_only)
        filename = f"{sequence:05d}-cue-{cue_id:05d}.txt"
        txt_path = output_dir / filename

        if skipped and not args.txt_include_skipped:
            skipped_count += 1
        else:
            txt_path.write_text(text + "\n", encoding="utf-8", newline="\n")
            written_count += 1

        rows.append(
            {
                "sequence": sequence,
                "cue_id": cue_id,
                "file": filename,
                "start_ms": int(cue["start_ms"]),
                "end_ms": int(cue["end_ms"]),
                "duration_ms": max(1, int(cue["end_ms"]) - int(cue["start_ms"])),
                "text": text,
                "skipped": skipped and not args.txt_include_skipped,
            }
        )

    manifest_path.write_text(
        json.dumps(
            {
                "source": str(srt_path),
                "output_dir": str(output_dir),
                "cue_count": len(cues),
                "txt_written": written_count,
                "txt_skipped": skipped_count,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Exported TXT files: {written_count}")
    print(f"Skipped cues: {skipped_count}")
    print(f"Output dir: {output_dir}")
    print(f"Manifest: {manifest_path}")


def command_assemble_audio(args: argparse.Namespace) -> None:
    ensure_ffmpeg_available()

    srt_path = Path(args.srt_file)
    audio_input_dir = Path(args.audio_input_dir) if args.audio_input_dir else default_txt_output_dir(srt_path)
    if not audio_input_dir.exists() or not audio_input_dir.is_dir():
        raise VoiceToolError(f"Audio input directory not found: {audio_input_dir}")

    manifest_path = Path(args.audio_manifest) if args.audio_manifest else audio_input_dir / "manifest.json"
    rows = load_audio_manifest(manifest_path)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise VoiceToolError(f"No rows found in manifest: {manifest_path}")

    output_path = Path(args.output) if args.output else default_assembled_output_path(srt_path)
    work_dir = Path(args.work_dir) if args.work_dir else default_voice_work_dir(srt_path)
    prepared_dir = work_dir / "assembled-prepared"
    fitted_dir = work_dir / "assembled-fitted"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    fitted_dir.mkdir(parents=True, exist_ok=True)

    final_end_ms = max(manifest_row_int(row, "end_ms", index) for index, row in enumerate(rows, start=1))
    timeline = AudioSegment.silent(duration=0, frame_rate=args.sample_rate).set_channels(2)
    cursor_ms = 0
    report: list[dict[str, Any]] = []

    print(f"Manifest rows: {len(rows)}")
    print(f"Audio input: {audio_input_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Output: {output_path}")

    for index, row in enumerate(rows, start=1):
        sequence = manifest_row_int(row, "sequence", index)
        cue_id = manifest_row_int(row, "cue_id", index)
        start_ms = manifest_row_int(row, "start_ms", index)
        end_ms = manifest_row_int(row, "end_ms", index)
        slot_ms = max(1, end_ms - start_ms)
        prefix = audio_prefix_from_manifest_row(row)
        warnings: list[str] = []

        if start_ms > cursor_ms:
            timeline += AudioSegment.silent(
                duration=start_ms - cursor_ms,
                frame_rate=args.sample_rate,
            ).set_channels(2)
            cursor_ms = start_ms
        elif start_ms < cursor_ms:
            warnings.append("cue_start_before_audio_cursor")

        audio_path = find_audio_for_manifest_row(audio_input_dir, row)
        if audio_path is None:
            if args.audio_missing == "error":
                raise VoiceToolError(
                    f"Missing audio for sequence {sequence}, cue {cue_id}, prefix {prefix!r} in {audio_input_dir}"
                )
            timeline += AudioSegment.silent(duration=slot_ms, frame_rate=args.sample_rate).set_channels(2)
            cursor_ms += slot_ms
            if cursor_ms > end_ms:
                warnings.append("audio_cursor_exceeds_cue_end")
            report.append(
                {
                    "sequence": sequence,
                    "cue_id": cue_id,
                    "source_audio": None,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "slot_ms": slot_ms,
                    "raw_ms": 0,
                    "required_speed": 0.0,
                    "applied_speed": 1.0,
                    "final_ms": slot_ms,
                    "trimmed": False,
                    "missing_audio": True,
                    "warning": "; ".join(warnings) if warnings else None,
                }
            )
            continue

        source_raw_ms = media_duration_ms(audio_path)
        print(f"[{index}/{len(rows)}] Assemble cue {cue_id}: {audio_path.name}")

        prepared_path = prepared_dir / (
            f"{index:05d}-cue-{cue_id:05d}-"
            f"silence-{args.silence_threshold_db}-{args.silence_min_len}-"
            f"{args.silence_keep_ms}.wav"
        )
        fitted_path = fitted_dir / f"{index:05d}-cue-{cue_id:05d}.wav"

        silence_info: dict[str, Any] = {"tts_silence_trimmed": False}
        fit_input_path = audio_path
        if args.trim_tts_silence:
            silence_info = trim_tts_silence(
                raw_path=audio_path,
                prepared_path=prepared_path,
                silence_thresh=args.silence_threshold_db,
                min_silence_len=args.silence_min_len,
                keep_silence_ms=args.silence_keep_ms,
                sample_rate=args.sample_rate,
            )
            fit_input_path = prepared_path

        fit_info = fit_audio_to_slot(
            raw_path=fit_input_path,
            processed_path=fitted_path,
            slot_ms=slot_ms,
            max_speed=args.max_speed,
            fit_mode=args.fit_mode,
            sample_rate=args.sample_rate,
        )

        segment = AudioSegment.from_file(fitted_path)
        if len(segment) > slot_ms and args.fit_mode == "trim":
            segment = segment[:slot_ms]
        timeline += segment
        cursor_ms += len(segment)

        if cursor_ms > end_ms:
            warnings.append("audio_cursor_exceeds_cue_end")
        if cursor_ms < end_ms:
            timeline += AudioSegment.silent(duration=end_ms - cursor_ms, frame_rate=args.sample_rate).set_channels(2)
            cursor_ms = end_ms

        report_row = {
            "sequence": sequence,
            "cue_id": cue_id,
            "source_audio": str(audio_path),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "slot_ms": slot_ms,
            "source_raw_ms": source_raw_ms,
            **silence_info,
            **fit_info,
            "final_ms": len(segment),
            "missing_audio": False,
            "warning": "; ".join(warnings) if warnings else None,
        }
        if fit_info["required_speed"] > args.max_speed:
            report_row["needs_shorter_text"] = True
        report.append(report_row)

    if cursor_ms < final_end_ms:
        timeline += AudioSegment.silent(duration=final_end_ms - cursor_ms, frame_rate=args.sample_rate).set_channels(2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    timeline.export(output_path, format=audio_export_format(output_path))

    report_path = Path(args.report) if args.report else output_path.with_suffix(".voice-report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    missing = sum(1 for row in report if row.get("missing_audio"))
    trimmed = sum(1 for row in report if row.get("trimmed"))
    warnings = sum(1 for row in report if row.get("warning"))
    print(f"Done: {output_path}")
    print(f"Report: {report_path}")
    print(f"Missing audio cues: {missing}")
    print(f"Trimmed cues: {trimmed}")
    print(f"Timeline warnings: {warnings}")


def command_build(args: argparse.Namespace) -> None:
    if args.scan_english_map:
        command_scan_english_map(args)
        return
    if args.export_txt:
        command_export_txt(args)
        return
    if args.assemble_audio:
        command_assemble_audio(args)
        return

    ensure_ffmpeg_available()

    srt_path = Path(args.srt_file)
    output_path = Path(args.output) if args.output else default_voice_output_path(srt_path)
    work_dir = Path(args.work_dir) if args.work_dir else default_voice_work_dir(srt_path)
    raw_dir = work_dir / "raw"
    prepared_dir = work_dir / "prepared"
    fitted_dir = work_dir / "fitted"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    fitted_dir.mkdir(parents=True, exist_ok=True)
    effective_voice = args.voice or DEFAULT_EDGE_VOICE
    raw_suffix = ".mp3"
    if args.phonetic_map_file:
        phonetic_map_path = Path(args.phonetic_map_file)
    else:
        phonetic_map_path = srt_path.with_name(f"{srt_path.stem}.english-map.py")
    if args.phonetic_english and phonetic_map_path.exists():
        loaded_count = load_phonetic_english_map(phonetic_map_path)
        print(f"Loaded phonetic map entries: {loaded_count} from {phonetic_map_path}")
    elif args.phonetic_english and args.phonetic_map_file:
        raise VoiceToolError(f"Phonetic map file not found: {phonetic_map_path}")

    cues = parse_srt(srt_path)
    if args.limit:
        cues = cues[: args.limit]
    if not cues:
        raise VoiceToolError(f"No cues found in {srt_path}")
    units = build_voice_units(
        cues,
        group_cues=args.group_cues,
        max_group_gap_ms=args.max_group_gap_ms,
        skip_bracket_only=args.skip_bracket_only,
    )

    final_end_ms = max(cue["end_ms"] for cue in cues)
    timeline = AudioSegment.silent(duration=0, frame_rate=args.sample_rate).set_channels(2)
    cursor_ms = 0
    report: list[dict[str, Any]] = []

    print(f"Cues: {len(cues)}")
    print(f"Voice units: {len(units)}")
    print(f"Group cues: {args.group_cues}")
    print("TTS backend: edge")
    print(f"Voice: {effective_voice}")
    print(f"Output: {output_path}")

    unit_index = 0
    while unit_index < len(units):
        unit = units[unit_index]
        unit_index += 1
        unit_id = unit["id"]
        start_ms = int(unit["start_ms"])
        end_ms = int(unit["end_ms"])
        slot_ms = max(1, end_ms - start_ms)
        text = str(unit["text"]).strip()
        tts_text = apply_phonetic_english(text) if args.phonetic_english else text

        if start_ms > cursor_ms:
            timeline += AudioSegment.silent(duration=start_ms - cursor_ms, frame_rate=args.sample_rate).set_channels(2)
            cursor_ms = start_ms
        elif start_ms < cursor_ms:
            report.append(
                {
                    "id": unit_id,
                    "ids": unit["ids"],
                    "warning": "unit_start_before_audio_cursor",
                    "start_ms": start_ms,
                    "cursor_ms": cursor_ms,
                }
            )

        if unit.get("skipped"):
            timeline += AudioSegment.silent(duration=slot_ms, frame_rate=args.sample_rate).set_channels(2)
            cursor_ms += slot_ms
            report.append({"id": unit_id, "ids": unit["ids"], "text": text, "skipped": True, "slot_ms": slot_ms})
            continue

        key = cache_key(
            f"edge:{args.mixed_english}:{args.phonetic_english}:{args.english_voice}:{tts_text}",
            effective_voice,
            args.rate,
            args.pitch,
            args.volume,
        )
        raw_path = raw_dir / f"{unit['file_id']}-{key}{raw_suffix}"
        prepared_path = prepared_dir / (
            f"{unit['file_id']}-{key}-"
            f"silence-{args.silence_threshold_db}-{args.silence_min_len}-"
            f"{args.silence_keep_ms}.wav"
        )
        fitted_path = fitted_dir / f"{unit['file_id']}-{key}.wav"

        if raw_path.exists() and raw_path.stat().st_size == 0:
            raw_path.unlink()

        if not raw_path.exists():
            print(f"[{unit_index}/{len(units)}] TTS unit {unit_id}")
            try:
                if args.mixed_english:
                    synthesize_edge_tts_mixed_with_retries(
                        text=text,
                        voice=effective_voice,
                        english_voice=args.english_voice,
                        output_path=raw_path,
                        rate=args.rate,
                        pitch=args.pitch,
                        volume=args.volume,
                        retries=args.tts_retries,
                    )
                else:
                    synthesize_edge_tts_with_retries(
                        text=tts_text,
                        voice=effective_voice,
                        output_path=raw_path,
                        rate=args.rate,
                        pitch=args.pitch,
                        volume=args.volume,
                        retries=args.tts_retries,
                    )
                if args.cue_delay > 0:
                    time.sleep(args.cue_delay)
            except VoiceToolError as exc:
                if raw_path.exists() and raw_path.stat().st_size == 0:
                    raw_path.unlink(missing_ok=True)
                if args.on_tts_fail == "split" and len(unit.get("cue_parts", [])) > 1:
                    split_units = [single_cue_unit(part) for part in unit["cue_parts"]]
                    units[unit_index:unit_index] = split_units
                    print(
                        f"Warning: unit {unit_id} TTS failed, splitting into {len(split_units)} cues: {exc}",
                        file=sys.stderr,
                    )
                    continue
                if args.on_tts_fail != "silence":
                    raise
                print(f"Warning: unit {unit_id} TTS failed, inserting silence: {exc}", file=sys.stderr)
                timeline += AudioSegment.silent(duration=slot_ms, frame_rate=args.sample_rate).set_channels(2)
                cursor_ms += slot_ms
                report.append(
                    {
                        "id": unit_id,
                        "ids": unit["ids"],
                        "text": text,
                        "tts_text": tts_text,
                        "slot_ms": slot_ms,
                        "tts_failed": True,
                        "error": str(exc),
                    }
                )
                continue
        else:
            print(f"[{unit_index}/{len(units)}] Cache unit {unit_id}")

        silence_info: dict[str, Any] = {"tts_silence_trimmed": False}
        fit_input_path = raw_path
        if args.trim_tts_silence:
            silence_info = trim_tts_silence(
                raw_path=raw_path,
                prepared_path=prepared_path,
                silence_thresh=args.silence_threshold_db,
                min_silence_len=args.silence_min_len,
                keep_silence_ms=args.silence_keep_ms,
                sample_rate=args.sample_rate,
            )
            fit_input_path = prepared_path

        fit_info = fit_audio_to_slot(
            raw_path=fit_input_path,
            processed_path=fitted_path,
            slot_ms=slot_ms,
            max_speed=args.max_speed,
            fit_mode=args.fit_mode,
            sample_rate=args.sample_rate,
        )

        segment = AudioSegment.from_file(fitted_path)
        if len(segment) > slot_ms and args.fit_mode == "trim":
            segment = segment[:slot_ms]
        timeline += segment
        cursor_ms += len(segment)

        if cursor_ms < end_ms:
            timeline += AudioSegment.silent(duration=end_ms - cursor_ms, frame_rate=args.sample_rate).set_channels(2)
            cursor_ms = end_ms

        report_row = {
            "id": unit_id,
            "ids": unit["ids"],
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": text,
            "tts_text": tts_text,
            **silence_info,
            **fit_info,
        }
        if fit_info["required_speed"] > args.max_speed:
            report_row["needs_shorter_text"] = True
        report.append(report_row)

    if cursor_ms < final_end_ms:
        timeline += AudioSegment.silent(duration=final_end_ms - cursor_ms, frame_rate=args.sample_rate).set_channels(2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    timeline.export(output_path, format=audio_export_format(output_path))

    report_path = Path(args.report) if args.report else output_path.with_suffix(".voice-report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    needs_shorter = sum(1 for row in report if row.get("needs_shorter_text"))
    trimmed = sum(1 for row in report if row.get("trimmed"))
    print(f"Done: {output_path}")
    print(f"Report: {report_path}")
    print(f"Cues needing shorter text: {needs_shorter}")
    print(f"Trimmed cues: {trimmed}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create timeline-aligned voice-over audio from translated SRT subtitles."
    )
    parser.add_argument("srt_file")
    parser.add_argument(
        "--scan-english-map",
        action="store_true",
        help="Scan the SRT for English/code terms and write a PHONETIC_ENGLISH_MAP template, then exit",
    )
    parser.add_argument(
        "--english-map-output",
        help="Output .py path for --scan-english-map, default is <input>.english-map.py",
    )
    parser.add_argument(
        "--auto-fill-english-map",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After scanning, call an OpenAI-compatible model to fill phonetic values automatically",
    )
    parser.add_argument(
        "--phonetic-map-model",
        default=DEFAULT_PHONETIC_MAP_MODEL,
        help="Model used by --auto-fill-english-map",
    )
    parser.add_argument(
        "--phonetic-map-base-url",
        default=DEFAULT_PHONETIC_MAP_BASE_URL,
        help="OpenAI-compatible API base URL used by --auto-fill-english-map",
    )
    parser.add_argument(
        "--phonetic-map-api-key-env",
        default="NINEROUTER_API_KEY",
        help="Env var containing API key for --auto-fill-english-map",
    )
    parser.add_argument(
        "--phonetic-map-timeout",
        type=int,
        default=180,
        help="HTTP timeout in seconds for --auto-fill-english-map",
    )
    parser.add_argument(
        "--phonetic-map-file",
        help="Load PHONETIC_ENGLISH_MAP_UPDATE from this .py file; default is <input>.english-map.py when --phonetic-english is used",
    )
    parser.add_argument(
        "--english-map-missing-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only write terms that are missing from PHONETIC_ENGLISH_MAP",
    )
    parser.add_argument(
        "--scan-lowercase-english",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also collect unknown lowercase ASCII words; this can include false positives",
    )
    parser.add_argument(
        "--export-txt",
        action="store_true",
        help="Export SRT cues as numbered .txt files for external batch TTS, then exit",
    )
    parser.add_argument(
        "--assemble-audio",
        action="store_true",
        help="Assemble externally generated cue audio files into one timeline-aligned voice track, then exit",
    )
    parser.add_argument(
        "--txt-output-dir",
        help="Output directory for --export-txt, default is <input>.txt-batch",
    )
    parser.add_argument(
        "--audio-input-dir",
        help="Directory containing external cue audio files, default is <input>.txt-batch",
    )
    parser.add_argument(
        "--audio-manifest",
        help="manifest.json path for --assemble-audio, default is <audio-input-dir>/manifest.json",
    )
    parser.add_argument(
        "--audio-missing",
        choices=["error", "silence"],
        default="error",
        help="For --assemble-audio: error stops on missing audio; silence inserts cue-length silence",
    )
    parser.add_argument(
        "--txt-include-skipped",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also write empty/bracket-only cues when exporting txt files",
    )
    parser.add_argument("--output", help="Output audio path, default is <input>.voice.wav or <input>.assembled.wav")
    parser.add_argument("--work-dir", help="Cache/work directory, default is <input folder>/voice_work")
    parser.add_argument(
        "--voice",
        help="Edge voice name. Default is vi-VN-NamMinhNeural.",
    )
    parser.add_argument("--rate", default="+0%", help="Edge TTS rate, for example +0%%, +10%%, -5%%")
    parser.add_argument("--pitch", default="+0Hz", help="Edge TTS pitch, for example +0Hz")
    parser.add_argument("--volume", default="+0%", help="Edge TTS volume, for example +0%%")
    parser.add_argument(
        "--mixed-english",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For Edge TTS, read known English/code terms with a separate English voice",
    )
    parser.add_argument(
        "--phonetic-english",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Read known English/code terms as Vietnamese phonetic text, for example website -> quep sai",
    )
    parser.add_argument(
        "--english-voice",
        default=DEFAULT_ENGLISH_VOICE,
        help="English Edge TTS voice used with --mixed-english",
    )
    parser.add_argument("--max-speed", type=float, default=1.25, help="Maximum ffmpeg speed-up factor")
    parser.add_argument(
        "--fit-mode",
        choices=["trim", "overflow"],
        default="trim",
        help="trim keeps timeline strict; overflow preserves all speech but can overlap later cues",
    )
    parser.add_argument(
        "--trim-tts-silence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trim leading/trailing silence from each TTS clip before fitting it into the timeline",
    )
    parser.add_argument(
        "--silence-threshold-db",
        type=int,
        default=-45,
        help="Silence threshold in dBFS used by --trim-tts-silence",
    )
    parser.add_argument(
        "--silence-min-len",
        type=int,
        default=120,
        help="Minimum silence length in ms used by --trim-tts-silence",
    )
    parser.add_argument(
        "--silence-keep-ms",
        type=int,
        default=60,
        help="Milliseconds of silence to keep at the start and end after trimming",
    )
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--limit", type=int, help="Process only first N cues for testing")
    parser.add_argument(
        "--group-cues",
        type=int,
        default=1,
        help="Group up to this many adjacent cues into one TTS request",
    )
    parser.add_argument(
        "--max-group-gap-ms",
        type=int,
        default=500,
        help="Maximum gap between adjacent cues that may be grouped",
    )
    parser.add_argument("--tts-retries", type=int, default=3, help="Retries per Edge TTS cue")
    parser.add_argument("--cue-delay", type=float, default=0.0, help="Seconds to wait after each new TTS request")
    parser.add_argument(
        "--on-tts-fail",
        choices=["error", "silence", "split"],
        default="error",
        help="error stops; silence inserts silence; split retries grouped units as single cues",
    )
    parser.add_argument("--report", help="Output JSON report path")
    parser.add_argument(
        "--skip-bracket-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip cues that are only [Music], (applause), etc.",
    )
    parser.set_defaults(func=command_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except VoiceToolError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

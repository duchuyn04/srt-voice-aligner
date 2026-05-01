#!/usr/bin/env python3
"""Local web app that turns uploaded video/audio into SRT subtitles.

The app extracts audio with ffmpeg, sends short chunks to OpenAI's audio
transcription API, merges timestamped segments back onto the source timeline,
and optionally calls the existing srt_translate.py pipeline to create a
Vietnamese SRT next to the original SRT.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_OUTPUT_DIR = "video_srt_outputs"
DEFAULT_UPLOAD_DIR = "video_srt_uploads"
DEFAULT_TRANSCRIBE_BASE_URL = "http://localhost:20128/v1"
DEFAULT_TRANSCRIBE_MODEL = "cx/gpt-5.5"
DEFAULT_TRANSLATE_BASE_URL = "http://localhost:20128/v1"
DEFAULT_TRANSLATE_MODEL = "cx/gpt-5.5"
DEFAULT_CHUNK_SECONDS = 600
DEFAULT_OVERLAP_SECONDS = 1
DEFAULT_MAX_CUE_MS = 6000
SAFE_STEM_RE = re.compile(r"[^\w .()\[\]-]+", re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


class VideoSrtError(RuntimeError):
    """Expected user-facing error."""


@dataclass
class Segment:
    start_ms: int
    end_ms: int
    text: str


@dataclass
class JobState:
    id: str
    filename: str
    status: str = "queued"
    message: str = "Queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    output_dir: str | None = None
    original_srt: str | None = None
    vi_srt: str | None = None
    report: str | None = None
    error: str | None = None
    progress: dict[str, Any] = field(default_factory=dict)

    def update(self, status: str | None = None, message: str | None = None, **progress: Any) -> None:
        if status is not None:
            self.status = status
        if message is not None:
            self.message = message
        if progress:
            self.progress.update(progress)
        self.updated_at = time.time()





def ms_to_srt_timestamp(value: int) -> str:
    value = max(0, int(round(value)))
    hours, remainder = divmod(value, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def safe_stem(value: str) -> str:
    stem = Path(value).stem.strip() or "video"
    stem = SAFE_STEM_RE.sub("_", stem)
    stem = re.sub(r"\s+", " ", stem).strip(" .")
    return stem or "video"


def ensure_tool(binary: str) -> None:
    try:
        subprocess.run(
            [binary, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise VideoSrtError(f"{binary} is required but was not found in PATH") from exc


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise VideoSrtError(f"Command failed: {' '.join(command)}\n{stderr}")
    return result


def media_duration_ms(path: Path) -> int:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    output = result.stdout.strip()
    if not output:
        raise VideoSrtError(f"Could not read media duration: {path}")
    return int(round(float(output) * 1000))


def extract_audio(source_path: Path, audio_path: Path) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio_path),
        ]
    )


def create_audio_chunks(
    audio_path: Path,
    chunks_dir: Path,
    duration_ms: int,
    chunk_seconds: int,
    overlap_seconds: int,
) -> list[dict[str, Any]]:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_ms = max(1, chunk_seconds * 1000)
    overlap_ms = max(0, min(overlap_seconds * 1000, chunk_ms - 1))
    chunks: list[dict[str, Any]] = []
    start_ms = 0
    index = 1

    while start_ms < duration_ms:
        chunk_duration_ms = min(chunk_ms, duration_ms - start_ms)
        chunk_path = chunks_dir / f"chunk-{index:04d}.mp3"
        run_command(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start_ms / 1000:.3f}",
                "-t",
                f"{chunk_duration_ms / 1000:.3f}",
                "-i",
                str(audio_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-b:a",
                "64k",
                str(chunk_path),
            ]
        )
        chunks.append({"index": index, "path": chunk_path, "offset_ms": start_ms, "duration_ms": chunk_duration_ms})
        index += 1
        if start_ms + chunk_ms >= duration_ms:
            break
        start_ms += chunk_ms - overlap_ms

    return chunks


def response_to_segments(response: Any, offset_ms: int) -> list[Segment]:
    raw_segments: Any = None
    if isinstance(response, dict):
        raw_segments = response.get("segments")
    else:
        raw_segments = getattr(response, "segments", None)

    if not raw_segments:
        text = response.get("text", "") if isinstance(response, dict) else getattr(response, "text", "")
        text = str(text).strip()
        if not text:
            return []
        return [Segment(start_ms=offset_ms, end_ms=offset_ms + 1000, text=text)]

    segments: list[Segment] = []
    for item in raw_segments:
        if isinstance(item, dict):
            start = item.get("start")
            end = item.get("end")
            text = item.get("text", "")
        else:
            start = getattr(item, "start", None)
            end = getattr(item, "end", None)
            text = getattr(item, "text", "")
        text = str(text).strip()
        if start is None or end is None or not text:
            continue
        start_ms = offset_ms + int(round(float(start) * 1000))
        end_ms = offset_ms + int(round(float(end) * 1000))
        if end_ms <= start_ms:
            end_ms = start_ms + 1
        segments.append(Segment(start_ms=start_ms, end_ms=end_ms, text=text))
    return segments


def transcribe_chunk(
    chunk_path: Path,
    offset_ms: int,
    model: str,
    base_url: str,
    retries: int = 3,
) -> list[Segment]:
    api_key = os.environ.get("OPENAI_API_KEY") or "local-no-key"
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise VideoSrtError("Missing Python package: openai. Install with: pip install openai") from exc

    client = OpenAI(api_key=api_key, base_url=base_url)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with chunk_path.open("rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model=model,
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )
            return response_to_segments(response, offset_ms=offset_ms)
        except Exception as exc:  # OpenAI SDK exposes several retryable exception types across versions.
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(2 ** attempt, 10))
    raise VideoSrtError(f"Transcription failed for {chunk_path.name}: {last_error}")


def merge_segments(chunks: list[list[Segment]], overlap_tolerance_ms: int = 250) -> list[Segment]:
    merged: list[Segment] = []
    cursor_ms = 0
    for chunk_segments in chunks:
        for segment in chunk_segments:
            text = segment.text.strip()
            if not text:
                continue
            if segment.end_ms <= cursor_ms + overlap_tolerance_ms:
                continue
            start_ms = max(segment.start_ms, cursor_ms)
            end_ms = max(segment.end_ms, start_ms + 1)
            merged.append(Segment(start_ms=start_ms, end_ms=end_ms, text=text))
            cursor_ms = end_ms
    return merged


def split_long_segment(segment: Segment, max_cue_ms: int) -> list[Segment]:
    duration = segment.end_ms - segment.start_ms
    if duration <= max_cue_ms:
        return [segment]
    parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(segment.text) if part.strip()]
    if len(parts) <= 1:
        words = segment.text.split()
        target_parts = max(2, int((duration + max_cue_ms - 1) // max_cue_ms))
        chunk_size = max(1, int((len(words) + target_parts - 1) // target_parts))
        parts = [" ".join(words[index : index + chunk_size]).strip() for index in range(0, len(words), chunk_size)]
        parts = [part for part in parts if part]
    if len(parts) <= 1:
        return [segment]

    total_chars = sum(len(part) for part in parts) or 1
    output: list[Segment] = []
    cursor = segment.start_ms
    for index, part in enumerate(parts):
        if index == len(parts) - 1:
            end_ms = segment.end_ms
        else:
            part_ms = max(1, int(round(duration * (len(part) / total_chars))))
            end_ms = min(segment.end_ms, cursor + part_ms)
        if end_ms <= cursor:
            end_ms = cursor + 1
        output.append(Segment(start_ms=cursor, end_ms=end_ms, text=part))
        cursor = end_ms
    return output


def normalize_segments_for_srt(segments: list[Segment], max_cue_ms: int = DEFAULT_MAX_CUE_MS) -> list[Segment]:
    normalized: list[Segment] = []
    cursor_ms = 0
    for segment in segments:
        for part in split_long_segment(segment, max_cue_ms=max_cue_ms):
            start_ms = max(part.start_ms, cursor_ms)
            end_ms = max(part.end_ms, start_ms + 1)
            text = part.text.strip()
            if not text:
                continue
            normalized.append(Segment(start_ms=start_ms, end_ms=end_ms, text=text))
            cursor_ms = end_ms
    return normalized


def write_srt(path: Path, segments: list[Segment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        text = segment.text.strip()
        if not text:
            continue
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{ms_to_srt_timestamp(segment.start_ms)} --> {ms_to_srt_timestamp(segment.end_ms)}",
                    text,
                ]
            )
        )
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8", newline="\n")


def translate_srt(
    source_srt: Path,
    output_srt: Path,
    work_dir: Path,
    base_url: str,
    model: str,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("srt_translate.py")),
        "all",
        str(source_srt),
        "--output",
        str(output_srt),
        "--work-dir",
        str(work_dir),
        "--base-url",
        base_url,
        "--models",
        model,
        "--skip-model-check",
    ]
    return run_command(command)


def write_report(job: JobState, report_path: Path, data: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"job": asdict(job), **data}
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cleanup_path(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def process_job(job: JobState, input_path: Path, settings: dict[str, Any]) -> None:
    temp_dir = input_path.parent
    output_root = Path(settings["output_dir"])
    report_data: dict[str, Any] = {"chunks": [], "translation": None}
    started_at = time.time()

    try:
        ensure_tool("ffmpeg")
        ensure_tool("ffprobe")
        stem = safe_stem(job.filename)
        output_dir = output_root / stem
        suffix = 1
        while output_dir.exists() and any(output_dir.iterdir()):
            output_dir = output_root / f"{stem}-{suffix}"
            suffix += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        job.output_dir = str(output_dir)
        original_srt = output_dir / f"{output_dir.name}.srt"
        vi_srt = output_dir / f"{output_dir.name}.vi.srt"
        report_path = output_dir / "job-report.json"
        job.original_srt = str(original_srt)
        job.vi_srt = str(vi_srt)
        job.report = str(report_path)

        job.update("running", "Extracting audio")
        audio_path = temp_dir / "audio.wav"
        extract_audio(input_path, audio_path)
        duration_ms = media_duration_ms(audio_path)
        report_data["duration_ms"] = duration_ms

        job.update("running", "Creating audio chunks")
        chunks = create_audio_chunks(
            audio_path=audio_path,
            chunks_dir=temp_dir / "chunks",
            duration_ms=duration_ms,
            chunk_seconds=int(settings["chunk_seconds"]),
            overlap_seconds=int(settings["overlap_seconds"]),
        )
        chunk_segments: list[list[Segment]] = []
        for chunk in chunks:
            job.update(
                "running",
                f"Transcribing chunk {chunk['index']}/{len(chunks)}",
                chunk=chunk["index"],
                chunks=len(chunks),
            )
            segments = transcribe_chunk(
                chunk_path=Path(chunk["path"]),
                offset_ms=int(chunk["offset_ms"]),
                model=str(settings["transcribe_model"]),
                base_url=str(settings["transcribe_base_url"]),
            )
            chunk_segments.append(segments)
            report_data["chunks"].append(
                {
                    "index": chunk["index"],
                    "offset_ms": chunk["offset_ms"],
                    "duration_ms": chunk["duration_ms"],
                    "segments": len(segments),
                }
            )

        job.update("running", "Writing original SRT")
        merged = merge_segments(chunk_segments)
        normalized = normalize_segments_for_srt(merged)
        write_srt(original_srt, normalized)
        report_data["segment_count"] = len(normalized)

        job.update("running", "Translating to Vietnamese")
        try:
            result = translate_srt(
                source_srt=original_srt,
                output_srt=vi_srt,
                work_dir=output_dir / "work",
                base_url=str(settings["translate_base_url"]),
                model=str(settings["translate_model"]),
            )
            report_data["translation"] = {"ok": True, "stdout": result.stdout, "stderr": result.stderr}
            job.update("done", "Done")
        except Exception as exc:
            report_data["translation"] = {"ok": False, "error": str(exc)}
            job.error = f"Vietnamese translation failed: {exc}"
            job.update("partial_done", "Original SRT is ready, Vietnamese translation failed")

        report_data["elapsed_seconds"] = round(time.time() - started_at, 2)
        write_report(job, report_path, report_data)
    except Exception as exc:
        job.error = str(exc)
        job.update("error", str(exc))
        if job.report:
            write_report(job, Path(job.report), {**report_data, "elapsed_seconds": round(time.time() - started_at, 2)})
    finally:
        cleanup_path(temp_dir)


JOBS: dict[str, JobState] = {}
APP_SETTINGS: dict[str, Any] = {}
app = FastAPI(title="Tao SRT tu Video")


INDEX_HTML = """
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tao SRT tu Video</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 32px; max-width: 980px; }}
    form {{ display: flex; gap: 12px; align-items: center; margin-bottom: 16px; flex-wrap: wrap; }}
    input[type=file], input[type=text] {{ flex: 1 1 360px; padding: 8px; }}
    button {{ padding: 8px 14px; cursor: pointer; }}
    button[disabled] {{ cursor: not-allowed; opacity: 0.6; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 24px; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
    code {{ background: #f2f2f2; padding: 2px 4px; }}
    .error {{ color: #b00020; }}
    .done {{ color: #0a7a32; }}
    .muted {{ color: #666; }}
  </style>
</head>
<body>
  <h1>Tao SRT tu Video</h1>
  <form id="upload-form">
    <input id="file-input" type="file" name="file">
    <button id="upload-button" type="submit">Tai len file da chon</button>
    <span id="pick-status" class="muted">Chua chon file</span>
  </form>
  <form id="path-form">
    <input id="path-input" type="text" placeholder="Hoac dan duong dan local, vi du C:\Videos\sample.mp4">
    <button id="path-button" type="submit">Dung duong dan local</button>
  </form>
  <p>Yeu cau: backend OpenAI-compatible local tai <code>http://localhost:20128/v1</code>, <code>ffmpeg</code>, <code>ffprobe</code>. Trang dung <code>fetch</code> de giu nguyen file da chon.</p>
  <table>
    <thead><tr><th>File</th><th>Status</th><th>Message</th><th>Downloads</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <script>
    const form = document.getElementById('upload-form');
    const input = document.getElementById('file-input');
    const button = document.getElementById('upload-button');
    const pickStatus = document.getElementById('pick-status');
    const pathForm = document.getElementById('path-form');
    const pathInput = document.getElementById('path-input');
    const pathButton = document.getElementById('path-button');
    input.addEventListener('change', () => {{
      if (!input.files || !input.files.length) {{
        pickStatus.textContent = 'Chua chon file';
        return;
      }}
      const file = input.files[0];
      pickStatus.textContent = `Da chon: ${{file.name}} (${{file.type || 'unknown'}}, ${{file.size}} bytes)`;
    }});

    form.addEventListener('submit', async (event) => {{
      event.preventDefault();
      if (!input.files || !input.files.length) {{
        pickStatus.textContent = 'Hay chon file truoc';
        return;
      }}
      const file = input.files[0];
      const data = new FormData();
      data.append('file', file);
      button.disabled = true;
      try {{
        const response = await fetch('/upload', {{ method: 'POST', body: data }});
        const raw = await response.text();
        if (!response.ok) {{
          pickStatus.textContent = `Loi tai len HTTP ${{response.status}}`;
          return;
        }}
        pickStatus.textContent = 'Tai len thanh cong. Dang tai lai de hien job moi...';
        setTimeout(() => location.reload(), 1200);
      }} catch (error) {{
        const message = error && error.message ? error.message : String(error);
        pickStatus.textContent = `Loi tai len: ${{message}}`;
      }} finally {{
        button.disabled = false;
      }}
    }});

    pathForm.addEventListener('submit', async (event) => {{
      event.preventDefault();
      const localPath = pathInput.value.trim();
      if (!localPath) {{
        return;
      }}
      pathButton.disabled = true;
      try {{
        const response = await fetch('/upload-local', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ path: localPath }})
        }});
        const raw = await response.text();
        if (!response.ok) {{
          pickStatus.textContent = `Loi duong dan local HTTP ${{response.status}}`;
          return;
        }}
        pickStatus.textContent = 'Da nhan duong dan local. Dang tai lai de hien job moi...';
        setTimeout(() => location.reload(), 1200);
      }} catch (error) {{
        const message = error && error.message ? error.message : String(error);
        pickStatus.textContent = `Loi duong dan local: ${{message}}`;
      }} finally {{
        pathButton.disabled = false;
      }}
    }});

  </script>
</body>
</html>
"""
def job_links(job: JobState) -> str:
    links: list[str] = [f'<a href="/jobs/{job.id}">JSON</a>']
    if job.original_srt and Path(job.original_srt).exists():
        links.append(f'<a href="/jobs/{job.id}/download/original-srt">SRT</a>')
    if job.vi_srt and Path(job.vi_srt).exists():
        links.append(f'<a href="/jobs/{job.id}/download/vi-srt">VI SRT</a>')
    if job.report and Path(job.report).exists():
        links.append(f'<a href="/jobs/{job.id}/download/report">Report</a>')
    return " | ".join(links)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    rows = []
    for job in sorted(JOBS.values(), key=lambda item: item.created_at, reverse=True):
        status_class = "error" if job.status == "error" else "done" if job.status in {"done", "partial_done"} else ""
        rows.append(
            f"<tr><td>{job.filename}</td><td class='{status_class}'>{job.status}</td>"
            f"<td>{job.message}</td><td>{job_links(job)}</td></tr>"
        )
    return INDEX_HTML.format(rows="\n".join(rows))


@app.post("/upload")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    job_id = uuid.uuid4().hex[:12]
    job = JobState(id=job_id, filename=file.filename)
    JOBS[job_id] = job

    upload_root = Path(APP_SETTINGS["upload_dir"])
    temp_dir = upload_root / job_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename).suffix or ".bin"
    input_path = temp_dir / f"input{suffix}"

    total_bytes = 0
    with input_path.open("wb") as handle:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total_bytes += len(chunk)
            handle.write(chunk)

    background_tasks.add_task(process_job, job, input_path, APP_SETTINGS.copy())
    return JSONResponse({"job_id": job_id, "status_url": f"/jobs/{job_id}", "filename": file.filename, "bytes": total_bytes})


@app.post("/upload-local")
async def upload_local(background_tasks: BackgroundTasks, payload: dict[str, Any]) -> JSONResponse:
    raw_path = str(payload.get("path") or "").strip()
    if not raw_path:
        raise HTTPException(status_code=400, detail="Missing path")
    source_path = Path(raw_path)
    if not source_path.exists() or not source_path.is_file():
        raise HTTPException(status_code=404, detail=f"Local file not found: {source_path}")

    job_id = uuid.uuid4().hex[:12]
    job = JobState(id=job_id, filename=source_path.name)
    JOBS[job_id] = job

    upload_root = Path(APP_SETTINGS["upload_dir"])
    temp_dir = upload_root / job_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    input_path = temp_dir / f"input{source_path.suffix or '.bin'}"
    shutil.copy2(source_path, input_path)
    background_tasks.add_task(process_job, job, input_path, APP_SETTINGS.copy())
    return JSONResponse({"job_id": job_id, "status_url": f"/jobs/{job_id}", "filename": source_path.name, "bytes": input_path.stat().st_size})


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return asdict(job)


def download_for(job_id: str, attr: str) -> FileResponse:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    value = getattr(job, attr)
    if not value or not Path(value).exists():
        raise HTTPException(status_code=404, detail="File is not ready")
    return FileResponse(Path(value), filename=Path(value).name)


@app.get("/jobs/{job_id}/download/original-srt")
def download_original_srt(job_id: str) -> FileResponse:
    return download_for(job_id, "original_srt")


@app.get("/jobs/{job_id}/download/vi-srt")
def download_vi_srt(job_id: str) -> FileResponse:
    return download_for(job_id, "vi_srt")


@app.get("/jobs/{job_id}/download/report")
def download_report(job_id: str) -> FileResponse:
    return download_for(job_id, "report")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local web app that creates SRT files from uploaded videos.")
    parser.add_argument("--host", default=os.environ.get("VIDEO_SRT_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VIDEO_SRT_PORT", DEFAULT_PORT)))
    parser.add_argument("--output-dir", default=os.environ.get("VIDEO_SRT_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    parser.add_argument("--upload-dir", default=os.environ.get("VIDEO_SRT_UPLOAD_DIR", DEFAULT_UPLOAD_DIR))
    parser.add_argument("--transcribe-base-url", default=os.environ.get("VIDEO_SRT_TRANSCRIBE_BASE_URL", DEFAULT_TRANSCRIBE_BASE_URL))
    parser.add_argument("--transcribe-model", default=os.environ.get("VIDEO_SRT_TRANSCRIBE_MODEL", DEFAULT_TRANSCRIBE_MODEL))
    parser.add_argument("--translate-base-url", default=os.environ.get("VIDEO_SRT_TRANSLATE_BASE_URL", DEFAULT_TRANSLATE_BASE_URL))
    parser.add_argument("--translate-model", default=os.environ.get("VIDEO_SRT_TRANSLATE_MODEL", DEFAULT_TRANSLATE_MODEL))
    parser.add_argument("--chunk-seconds", type=int, default=int(os.environ.get("VIDEO_SRT_CHUNK_SECONDS", DEFAULT_CHUNK_SECONDS)))
    parser.add_argument("--overlap-seconds", type=int, default=int(os.environ.get("VIDEO_SRT_OVERLAP_SECONDS", DEFAULT_OVERLAP_SECONDS)))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    APP_SETTINGS.update(vars(args))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.upload_dir).mkdir(parents=True, exist_ok=True)
    print(f"Tao SRT tu Video web app: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())









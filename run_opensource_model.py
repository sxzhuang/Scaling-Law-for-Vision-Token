import argparse
import asyncio
import base64
import csv
import logging
import mimetypes
import random
import re
from pathlib import Path

import httpx
from openai import AsyncOpenAI


CLEAN_FENCE_HEAD = re.compile(r"^```[a-zA-Z]*\n")
CLEAN_FENCE_TAIL = re.compile(r"\n```$")
URL_IMAGE_PREFIX = "http://127.0.0.1:18080"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concurrent OCR via OpenAI-compatible vLLM server.")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API key for the OpenAI-compatible endpoint.")
    parser.add_argument("--url", type=str, default="http://10.74.197.104:50016/v1", help="OpenAI-compatible base URL.")
    parser.add_argument("--input_path", type=Path, required=True, help="Directory containing .jpg images.")
    parser.add_argument("--output_path", type=Path, required=True, help="Directory to save OCR markdown files.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model name.")
    parser.add_argument("--timeout", type=float, default=300, help="Request timeout in seconds.")
    parser.add_argument("--max_tokens", type=int, default=6500, help="Maximum completion tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--concurrency", type=int, default=8, help="Max in-flight requests.")
    parser.add_argument("--retries", type=int, default=5, help="Retry count for failed requests.")
    parser.add_argument("--image_url_mode", choices=["file", "data", "url"], default="file", help="file: send file:// URL (fast). data: send base64 data URL (slow but universal). url: send URL built as http://127.0.0.1:18080/<filename>.")
    return parser.parse_args()


def encode_image_to_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type:
        mime_type = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def build_messages(prompt: str, image_path: Path, image_url_mode: str, url_prefix: str | None = None) -> list[dict]:
    if image_url_mode == "file":
        image_url = image_path.resolve().as_uri()
    elif image_url_mode == "data":
        image_url = encode_image_to_data_url(image_path)
    else:
        if not url_prefix:
            url_prefix = URL_IMAGE_PREFIX
        image_url = f"{url_prefix.rstrip('/')}/{image_path.name}"
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]


def clean_markdown(content: str) -> str:
    text = content.strip()
    text = CLEAN_FENCE_HEAD.sub("", text)
    text = CLEAN_FENCE_TAIL.sub("", text)
    return text.strip()


def load_existing_records(records_path: Path) -> list[list[str]]:
    if not records_path.exists():
        return []
    rows: list[list[str]] = []
    with records_path.open("r", newline="", encoding="utf-8") as file_handle:
        reader = csv.reader(file_handle)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            image_id = row[0]
            finish_reason = row[1] if len(row) > 1 else ""
            if image_id:
                rows.append([image_id, finish_reason])
    return rows


async def ocr_one(
    client: AsyncOpenAI,
    prompt: str,
    image_path: Path,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    url_prefix: str | None,
) -> tuple[str, str, Path | None]:
    image_id = image_path.stem
    last_exception: Exception | None = None

    for attempt in range(1, args.retries + 1):
        try:
            async with semaphore:
                if args.image_url_mode == "data":
                    messages = await asyncio.to_thread(build_messages, prompt, image_path, "data", url_prefix)
                else:
                    messages = build_messages(prompt, image_path, args.image_url_mode, url_prefix)
                response = await client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    timeout=args.timeout,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
            if not response or not response.choices:
                raise RuntimeError("Empty response choices")
            choice = response.choices[0]
            finish_reason = choice.finish_reason or "stop"
            content = choice.message.content if choice.message else ""
            cleaned = clean_markdown(content)
            output_file = args.output_path / f"{image_id}.md"
            output_file.write_text(cleaned, encoding="utf-8")
            recorded_reason = "length" if finish_reason == "length" else "stop"
            return image_id, recorded_reason, output_file
        except Exception as exc:
            last_exception = exc
            backoff = min(2.0 ** (attempt - 1), 8.0) + random.uniform(0, 0.3)
            logging.warning("Attempt %d failed for %s: %s (sleep %.2fs)", attempt, image_path, exc, backoff)
            await asyncio.sleep(backoff)

    logging.error("Failed OCR after retries for %s: %s", image_path, last_exception)
    return image_id, "fail", None


async def amain() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    prompt = "You are a high-fidelity OCR (Optical Character Recognition) to Markdown converter. Your goal is to transcribe the content of the provided image into Markdown format with pixel-perfect accuracy regarding text and structure. Convert the visible text and structural elements in the image into raw Markdown code. # Output Format (CRITICAL) \n You must output the final result wrapped inside a Markdown code block. Your response must look exactly like this structure:```markdown \n (The transcribed content goes here) \n```"
    if not args.input_path.is_dir():
        logging.error("Input path is not a directory: %s", args.input_path)
        return
    args.output_path.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(args.input_path.glob("*.jpg"))
    if not image_paths:
        logging.warning("No .jpg images found in %s", args.input_path)
        return

    records_path = args.output_path.parent / "generation_records.csv"
    existing_rows = load_existing_records(records_path)
    existing_ids = {row[0] for row in existing_rows}
    finish_records: dict[str, str] = {}
    pending_paths: list[Path] = []
    skipped_ids: list[str] = []
    for image_path in image_paths:
        output_file = args.output_path / f"{image_path.stem}.md"
        if output_file.exists():
            skipped_ids.append(image_path.stem)
            logging.info("Skipped existing OCR output: %s", output_file)
        else:
            pending_paths.append(image_path)

    url_prefix = None
    if args.image_url_mode == "url":
        url_prefix = f"{URL_IMAGE_PREFIX.rstrip('/')}/{args.input_path.as_posix().lstrip('/')}"

    if pending_paths:
        limits = httpx.Limits(
            max_connections=max(32, args.concurrency * 2),
            max_keepalive_connections=max(16, args.concurrency),
            keepalive_expiry=60.0,
        )
        http_client = httpx.AsyncClient(limits=limits)
        client = AsyncOpenAI(api_key=args.api_key, base_url=args.url, http_client=http_client)
        semaphore = asyncio.Semaphore(args.concurrency)
        tasks = [asyncio.create_task(ocr_one(client, prompt, image_path, args, semaphore, url_prefix)) for image_path in pending_paths]
        try:
            for coro in asyncio.as_completed(tasks):
                image_id, finish_reason, output_file = await coro
                finish_records[image_id] = finish_reason
                if output_file:
                    logging.info("Saved OCR output: %s", output_file)
                else:
                    logging.error("No OCR output for %s", image_id)
        finally:
            await http_client.aclose()

    for row in existing_rows:
        image_id = row[0]
        if image_id in finish_records:
            row[1] = finish_records[image_id]
    for image_id, finish_reason in finish_records.items():
        if image_id not in existing_ids:
            existing_rows.append([image_id, finish_reason])

    fail_count = sum(1 for finish_reason in finish_records.values() if finish_reason == "fail")
    logging.info("Fail count for this run: %d", fail_count)

    with records_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["image_id", "finish_reason"])
        writer.writerows(existing_rows)
    logging.info("Saved generation records: %s", records_path)


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()

import argparse
import asyncio
import base64
import csv
import json
import logging
import math
import mimetypes
import random
import re
from pathlib import Path

import httpx
from openai import AsyncOpenAI


URL_IMAGE_PREFIX = "http://127.0.0.1:18080"
OCR_PROMPT = "Extract all readable text from the image. Return only the transcribed text as continuous plain text. Preserve the reading order, but do not preserve visual line breaks. Do not add explanations, comments, markdown, or formatting."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR with an OpenAI-compatible vision-language model server.")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API key for the OpenAI-compatible endpoint.")
    parser.add_argument("--url", type=str, default="http://10.74.197.104:50016/v1", help="OpenAI-compatible base URL.")
    parser.add_argument("--input_path", type=Path, required=True, help="Directory containing .jpg images.")
    parser.add_argument("--output_path", type=Path, required=True, help="Directory to save OCR prediction markdown files.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model name served by vLLM.")
    parser.add_argument("--timeout", type=float, default=300, help="Request timeout in seconds.")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Maximum completion tokens.")
    parser.add_argument("--metadata_json", type=Path, default=None, help="Optional JSON file containing per-image GT text.")
    parser.add_argument("--metadata_text_field", type=str, default="context", help="Field name in metadata_json used as GT OCR text.")
    parser.add_argument("--metadata_id_field", type=str, default="id", help="Field name in metadata_json used to match image stems.")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Tokenizer used to estimate GT token lengths. Defaults to --model.")
    parser.add_argument("--max_tokens_ratio", type=float, default=1.2, help="Multiplier applied to GT token length for per-sample max_tokens.")
    parser.add_argument("--max_tokens_margin", type=int, default=32, help="Extra token buffer added after max_tokens_ratio.")
    parser.add_argument("--min_max_tokens", type=int, default=64, help="Minimum per-sample max_tokens when dynamic limits are enabled.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--concurrency", type=int, default=8, help="Max in-flight requests.")
    parser.add_argument("--retries", type=int, default=5, help="Retry count for failed requests.")
    parser.add_argument("--image_url_mode", choices=["file", "data", "url"], default="file", help="file: send file:// URL. data: send base64 data URL. url: send URL built as http://127.0.0.1:18080/<path>/<filename>.")
    parser.add_argument("--prompt", type=str, default=OCR_PROMPT, help="OCR prompt sent with each image.")
    return parser.parse_args()


def encode_image_to_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type:
        mime_type = "image/jpeg"
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


def normalize_ocr_output(raw: str) -> str:
    return re.sub(r"-\s*\n\s*", "-", raw).strip()


def load_dynamic_max_tokens(args: argparse.Namespace) -> dict[str, int]:
    if args.metadata_json is None:
        return {}
    if not args.metadata_json.exists():
        logging.warning("Metadata JSON not found, using global max_tokens: %s", args.metadata_json)
        return {}

    tokenizer_name = args.tokenizer_name or args.model
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    except Exception as exc:
        logging.warning("Failed to load tokenizer %s, using global max_tokens: %s", tokenizer_name, exc)
        return {}

    with args.metadata_json.open("r", encoding="utf-8") as file_handle:
        records = json.load(file_handle)
    if not isinstance(records, list):
        logging.warning("Metadata JSON must contain a list, using global max_tokens: %s", args.metadata_json)
        return {}

    max_tokens_by_id = {}
    skipped = 0
    for record in records:
        if not isinstance(record, dict):
            skipped += 1
            continue
        image_id = str(record.get(args.metadata_id_field, ""))
        gt_text = record.get(args.metadata_text_field)
        if not image_id or not isinstance(gt_text, str):
            skipped += 1
            continue
        token_ids = tokenizer(gt_text, add_special_tokens=False, truncation=False).get("input_ids", [])
        dynamic_max_tokens = math.ceil(len(token_ids) * args.max_tokens_ratio) + args.max_tokens_margin
        max_tokens_by_id[image_id] = min(args.max_tokens, max(args.min_max_tokens, dynamic_max_tokens))

    logging.info("Loaded dynamic max_tokens for %d records from %s", len(max_tokens_by_id), args.metadata_json)
    if skipped:
        logging.warning("Skipped %d metadata records without usable id/text fields", skipped)
    return max_tokens_by_id


def load_existing_records(records_path: Path) -> list[list[str]]:
    if not records_path.exists():
        return []
    rows = []
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


async def ocr_one(client: AsyncOpenAI, image_path: Path, args: argparse.Namespace, semaphore: asyncio.Semaphore, url_prefix: str | None, max_tokens_by_id: dict[str, int]) -> tuple[str, str, Path | None, dict | None, int]:
    image_id = image_path.stem
    request_max_tokens = max_tokens_by_id.get(image_id, args.max_tokens)
    last_exception: Exception | None = None

    for attempt in range(1, args.retries + 1):
        try:
            async with semaphore:
                if args.image_url_mode == "data":
                    messages = await asyncio.to_thread(build_messages, args.prompt, image_path, "data", url_prefix)
                else:
                    messages = build_messages(args.prompt, image_path, args.image_url_mode, url_prefix)
                response = await client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    timeout=args.timeout,
                    max_tokens=request_max_tokens,
                    temperature=args.temperature,
                )

            if not response or not response.choices:
                raise RuntimeError("Empty response choices")

            choice = response.choices[0]
            finish_reason = choice.finish_reason or "stop"
            raw_prediction = choice.message.content if choice.message and choice.message.content else ""
            prediction = normalize_ocr_output(raw_prediction)
            output_file = args.output_path / f"{image_id}.md"
            output_file.write_text(prediction + "\n", encoding="utf-8")
            result = {
                "id": image_id,
                "image": image_path.name,
                "prediction": prediction,
                "finish_reason": finish_reason,
                "max_tokens": request_max_tokens,
            }
            recorded_reason = "length" if finish_reason == "length" else "stop"
            return image_id, recorded_reason, output_file, result, request_max_tokens
        except Exception as exc:
            last_exception = exc
            backoff = min(2.0 ** (attempt - 1), 8.0) + random.uniform(0, 0.3)
            logging.warning("Attempt %d failed for %s: %s (sleep %.2fs)", attempt, image_path, exc, backoff)
            await asyncio.sleep(backoff)

    logging.error("Failed OCR after retries for %s: %s", image_path, last_exception)
    return image_id, "fail", None, None, request_max_tokens


async def amain() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.input_path.is_dir():
        logging.error("Input path is not a directory: %s", args.input_path)
        return

    args.output_path.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(args.input_path.glob("*.jpg"))
    if not image_paths:
        logging.warning("No .jpg images found in %s", args.input_path)
        return
    max_tokens_by_id = load_dynamic_max_tokens(args)

    output_suffix = args.output_path.name
    records_path = args.output_path.parent / f"generation_records_{output_suffix}.csv"
    existing_rows = load_existing_records(records_path)
    existing_ids = {row[0] for row in existing_rows}
    finish_records: dict[str, str] = {}
    pending_paths = []
    for image_path in image_paths:
        output_file = args.output_path / f"{image_path.stem}.md"
        if output_file.exists():
            logging.info("Skipped existing OCR output: %s", output_file)
        else:
            pending_paths.append(image_path)

    url_prefix = None
    if args.image_url_mode == "url":
        url_prefix = f"{URL_IMAGE_PREFIX.rstrip('/')}/{args.input_path.as_posix().lstrip('/')}"

    results = []
    if pending_paths:
        limits = httpx.Limits(
            max_connections=max(32, args.concurrency * 2),
            max_keepalive_connections=max(16, args.concurrency),
            keepalive_expiry=60.0,
        )
        http_client = httpx.AsyncClient(limits=limits)
        client = AsyncOpenAI(api_key=args.api_key, base_url=args.url, http_client=http_client)
        semaphore = asyncio.Semaphore(args.concurrency)
        tasks = [asyncio.create_task(ocr_one(client, image_path, args, semaphore, url_prefix, max_tokens_by_id)) for image_path in pending_paths]
        try:
            for coro in asyncio.as_completed(tasks):
                image_id, finish_reason, output_file, result, request_max_tokens = await coro
                finish_records[image_id] = finish_reason
                if output_file:
                    logging.info("Saved OCR output: %s (finish_reason=%s, max_tokens=%d)", output_file, finish_reason, request_max_tokens)
                else:
                    logging.error("No OCR output for %s (max_tokens=%d)", image_id, request_max_tokens)
                if result is not None:
                    results.append(result)
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

    if results:
        results_path = args.output_path.parent / f"ocr_predictions_{output_suffix}.jsonl"
        with results_path.open("a", encoding="utf-8") as f:
            for result in sorted(results, key=lambda item: item["id"]):
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        logging.info("Appended %d OCR prediction records to %s", len(results), results_path)


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()

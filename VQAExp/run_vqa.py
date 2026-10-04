import argparse
import asyncio
import base64
import csv
import json
import logging
import mimetypes
import random
from pathlib import Path

import httpx
from openai import AsyncOpenAI


URL_IMAGE_PREFIX = "http://127.0.0.1:18080"
RETRIEVAL_PROMPT_PREFIX = "Answer a question based on the above book snippet. Your answer should be short and based on either explicitly stated facts or strong, logical inferences. Return only the final answer with no additional explanation or reasoning. Question: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VTC-Retrieval VQA with an OpenAI-compatible vLLM server.")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API key for the OpenAI-compatible endpoint.")
    parser.add_argument("--url", type=str, default="http://10.74.197.104:50016/v1", help="OpenAI-compatible base URL.")
    parser.add_argument("--input_path", type=Path, required=True, help="Directory containing .jpg images.")
    parser.add_argument("--dataset_json", type=Path, default=Path("VQAExp/data/Retrieval/retrieval_cropped_0_25k_600.json"), help="JSON dataset containing id, problem, and answers fields.")
    parser.add_argument("--output_path", type=Path, required=True, help="Directory to save VQA prediction text files.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model name served by vLLM.")
    parser.add_argument("--timeout", type=float, default=300, help="Request timeout in seconds.")
    parser.add_argument("--max_tokens", type=int, default=128, help="Maximum completion tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--concurrency", type=int, default=8, help="Max in-flight requests.")
    parser.add_argument("--retries", type=int, default=5, help="Retry count for failed requests.")
    parser.add_argument("--image_url_mode", choices=["file", "data", "url"], default="file", help="file: send file:// URL. data: send base64 data URL. url: send URL built as http://127.0.0.1:18080/<path>/<filename>.")
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


def load_dataset_by_id(dataset_json: Path) -> dict[str, dict]:
    with dataset_json.open("r", encoding="utf-8") as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        raise ValueError("dataset_json must contain a list of samples.")

    by_id = {}
    for sample in samples:
        sample_id = str(sample.get("id", "")).strip()
        if not sample_id:
            continue
        if "problem" not in sample:
            raise ValueError(f"Sample {sample_id} is missing problem field.")
        by_id[sample_id] = sample
    return by_id


def build_prompt(problem: str) -> str:
    return f"{RETRIEVAL_PROMPT_PREFIX}{problem}"


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


async def vqa_one(
    client: AsyncOpenAI,
    image_path: Path,
    sample: dict,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    url_prefix: str | None,
) -> tuple[str, str, Path | None, dict | None]:
    image_id = image_path.stem
    prompt = build_prompt(str(sample["problem"]))
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
            prediction = choice.message.content.strip() if choice.message and choice.message.content else ""
            output_file = args.output_path / f"{image_id}.txt"
            output_file.write_text(prediction + "\n", encoding="utf-8")
            result = {
                "id": image_id,
                "image": image_path.name,
                "problem": sample["problem"],
                "answers": sample.get("answers", []),
                "prediction": prediction,
                "finish_reason": finish_reason,
            }
            recorded_reason = "length" if finish_reason == "length" else "stop"
            return image_id, recorded_reason, output_file, result
        except Exception as exc:
            last_exception = exc
            backoff = min(2.0 ** (attempt - 1), 8.0) + random.uniform(0, 0.3)
            logging.warning("Attempt %d failed for %s: %s (sleep %.2fs)", attempt, image_path, exc, backoff)
            await asyncio.sleep(backoff)

    logging.error("Failed VQA after retries for %s: %s", image_path, last_exception)
    return image_id, "fail", None, None


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

    dataset_by_id = load_dataset_by_id(args.dataset_json)
    image_sample_pairs = []
    missing_ids = []
    for image_path in image_paths:
        sample = dataset_by_id.get(image_path.stem)
        if sample is None:
            missing_ids.append(image_path.stem)
            continue
        image_sample_pairs.append((image_path, sample))

    if missing_ids:
        logging.warning("Missing JSON records for %d images, e.g. %s", len(missing_ids), missing_ids[:5])
    if not image_sample_pairs:
        logging.error("No images have matching dataset_json records.")
        return

    output_suffix = args.output_path.name
    records_path = args.output_path.parent / f"generation_records_{output_suffix}.csv"
    existing_rows = load_existing_records(records_path)
    existing_ids = {row[0] for row in existing_rows}
    finish_records: dict[str, str] = {}
    pending_pairs = []
    for image_path, sample in image_sample_pairs:
        output_file = args.output_path / f"{image_path.stem}.txt"
        if output_file.exists():
            logging.info("Skipped existing VQA output: %s", output_file)
        else:
            pending_pairs.append((image_path, sample))

    url_prefix = None
    if args.image_url_mode == "url":
        url_prefix = f"{URL_IMAGE_PREFIX.rstrip('/')}/{args.input_path.as_posix().lstrip('/')}"

    results = []
    if pending_pairs:
        limits = httpx.Limits(
            max_connections=max(32, args.concurrency * 2),
            max_keepalive_connections=max(16, args.concurrency),
            keepalive_expiry=60.0,
        )
        http_client = httpx.AsyncClient(limits=limits)
        client = AsyncOpenAI(api_key=args.api_key, base_url=args.url, http_client=http_client)
        semaphore = asyncio.Semaphore(args.concurrency)
        tasks = [asyncio.create_task(vqa_one(client, image_path, sample, args, semaphore, url_prefix)) for image_path, sample in pending_pairs]
        try:
            for coro in asyncio.as_completed(tasks):
                image_id, finish_reason, output_file, result = await coro
                finish_records[image_id] = finish_reason
                if output_file:
                    logging.info("Saved VQA output: %s", output_file)
                else:
                    logging.error("No VQA output for %s", image_id)
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
        results_path = args.output_path.parent / f"vqa_predictions_{output_suffix}.jsonl"
        with results_path.open("a", encoding="utf-8") as f:
            for result in sorted(results, key=lambda item: item["id"]):
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        logging.info("Appended %d prediction records to %s", len(results), results_path)


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()

# python VQAExp/run_vqa.py \
#     --input_path VQAExp/data/Retrieval/28_6_0/images_640 \
#     --dataset_json VQAExp/data/Retrieval/retrieval_cropped_0_25k_600.json \
#     --output_path VQAExp/vqa_predictions/Qwen/Qwen2.5-VL-7B-Instruct/Retrieval/28_6_0/res_640 \
#     --url http://localhost:50016/v1 \
#     --model Qwen/Qwen2.5-VL-7B-Instruct \
#     --max_tokens 128 \
#     --image_url_mode file \
#     --concurrency 8

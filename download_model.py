from huggingface_hub import snapshot_download
import time

def download_with_retry(repo_id, local_dir=None, max_retries=5):
    for attempt in range(max_retries):
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                resume_download=True,
                max_workers=4,
            )
            print(f"✓ Download completed successfully")
            if local_dir:
                print(f"Model saved to: {local_dir}")
            else:
                print(f"Model saved to default cache: ~/.cache/huggingface/hub/")
            return
        except Exception as e:
            print(f"✗ Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt * 10
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise

if __name__ == "__main__":
    model_id = "zai-org/Glyph"
    download_with_retry(
        repo_id=model_id,
        local_dir=None
    )
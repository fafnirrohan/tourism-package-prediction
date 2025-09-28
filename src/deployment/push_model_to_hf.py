#!/usr/bin/env python3
import argparse, os, sys, logging, traceback
from huggingface_hub import HfApi
logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

def try_create(api, repo_id, token):
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", token=token, exist_ok=True)
        return True
    except Exception as e:
        log.warning("create_repo failed: %s", e)
        return False

def try_upload(api, model_file, repo_id, token):
    try:
        api.upload_file(path_or_fileobj=model_file, path_in_repo=model_file.split("/")[-1], repo_id=repo_id, token=token)
        return True
    except Exception as e:
        log.error("upload failed: %s", e)
        traceback.print_exc()
        return False

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--token", default=None)
    args = p.parse_args()
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        log.error("HF token missing")
        sys.exit(2)
    model_file = args.model_path
    if not os.path.exists(model_file):
        log.error("Model file missing: %s", model_file)
        sys.exit(2)
    api = HfApi()
    ok = try_create(api, args.repo_id, token)
    if ok:
        u = try_upload(api, model_file, args.repo_id, token)
        if u:
            print("Uploaded model to", args.repo_id)
            sys.exit(0)
    # fallback: try upload directly (if repo exists)
    if try_upload(api, model_file, args.repo_id, token):
        print("Uploaded model to", args.repo_id)
        sys.exit(0)
    log.error("Failed to upload; please ensure repo exists and token has write permissions")
    sys.exit(1)

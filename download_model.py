from huggingface_hub import hf_hub_download
import shutil
import os

repo_id = "ekwek/Soprano-80M"
files = ["tokenizer.json", "config.json", "model.safetensors"]

for file in files:
    print(f"Downloading {file}...")
    path = hf_hub_download(repo_id=repo_id, filename=file)
    print(f"Downloaded to {path}")
    # Copy to current directory
    dest = file
    print(f"Copying to {dest}...")
    shutil.copy(path, dest)

print("Done.")

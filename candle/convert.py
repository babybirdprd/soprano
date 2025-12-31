import torch
import json
import os
import shutil
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download

def convert():
    print("Downloading model files...")
    repo_id = "ekwek/Soprano-80M"

    # Ensure candle directory exists
    os.makedirs("candle", exist_ok=True)

    # Download decoder
    decoder_path = hf_hub_download(repo_id=repo_id, filename="decoder.pth")

    # Download Qwen files
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
    model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")

    print("Copying Qwen files to candle/ directory...")
    shutil.copy(config_path, "candle/config.json")
    shutil.copy(tokenizer_path, "candle/tokenizer.json")
    # model.safetensors might be large, maybe symlink or copy? Copy for safety.
    shutil.copy(model_path, "candle/model.safetensors")

    print("Loading decoder weights...")
    state_dict = torch.load(decoder_path, map_location="cpu")

    print("Saving decoder weights to candle/decoder.safetensors...")
    save_file(state_dict, "candle/decoder.safetensors")

    print("Done.")

if __name__ == "__main__":
    convert()

import torch
import json
import os
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download

def convert():
    print("Downloading model...")
    repo_id = "ekwek/Soprano-80M"
    decoder_path = hf_hub_download(repo_id=repo_id, filename="decoder.pth")
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")

    print("Loading weights...")
    state_dict = torch.load(decoder_path, map_location="cpu")

    # Extract config
    # The config.json in the repo is likely for Qwen.
    # The Vocos config might be hardcoded in the Python code or passed in arguments.
    # Looking at tts.py (not read yet) or decoder.py might reveal how it's initialized.
    # But decoder.pth only contains weights.

    # Wait, the prompt says "Extract the Vocos configuration (dim, layers, etc.) into a JSON file".
    # Since I don't see a vocos_config.json in the repo file list (I haven't listed repo files via tool, just guessed),
    # I should check how `decoder` is initialized in `soprano/tts.py`.

    print("Saving weights to decoder.safetensors...")
    save_file(state_dict, "candle/decoder.safetensors")

    print("Done.")

if __name__ == "__main__":
    convert()

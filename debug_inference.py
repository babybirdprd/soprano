import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file
import os

def debug_inference():
    prompt_text = "hello world!"
    # The cleaned prompt logic from Rust: Format is [STOP][TEXT]...[START]
    # In Rust: let prompt = format!("[STOP][TEXT]{}[START]", cleaned_prompt);
    full_prompt = f"[STOP][TEXT]{prompt_text}[START]"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("d:/soprano/candle", trust_remote_code=True)
    inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"]
    
    print(f"Prompt tokens: {input_ids[0].tolist()}")
    
    # Load Qwen model
    model = AutoModelForCausalLM.from_pretrained("d:/soprano/candle", trust_remote_code=True, torch_dtype=torch.float32).to("cpu")
    model.eval()
    
    generated_tokens = []
    generated_hidden_states = []
    
    current_input = input_ids
    past_key_values = None
    
    print("Generating tokens (greedy)...")
    for i in range(50):
        with torch.no_grad():
            outputs = model(current_input, past_key_values=past_key_values, output_hidden_states=True, use_cache=True)
            
        logits = outputs.logits[:, -1, :]
        hidden_states = outputs.hidden_states[-1][:, -1, :] # Last layer, last token
        
        next_token = torch.argmax(logits, dim=-1)
        
        if next_token.item() == 3: # [STOP]
            break
            
        generated_tokens.append(next_token.item())
        generated_hidden_states.append(hidden_states.squeeze().cpu().numpy())
        
        current_input = next_token.unsqueeze(0)
        past_key_values = outputs.past_key_values
        
    print(f"Generated tokens: {generated_tokens}")
    
    # Dump first few hidden states for comparison
    if generated_hidden_states:
        hs0 = generated_hidden_states[0]
        print(f"PY_HS0 head: {hs0[:10].tolist()}")
        print(f"PY_HS0 range: [{hs0.min()}, {hs0.max()}]")
        
    # Also dump input hidden states (before generation starts)
    # The last hidden state of the prompt is what starts the audio?
    # Actually, Soprano extracts hidden states for all tokens generated AFTER [START]
    # but does it include the [START] token's hidden state?
    # In my Rust main.rs:
    # it loop 256 times
    # in each loop: forward, collect HS of the input, then get next_token.
    # So it collects hidden states of tokens that are *input* to the transformer.
    
    # Wait, my Rust logic:
    # for iter in 0..256 {
    #   (logits, hidden_states) = qwen.forward(&current_input, pos)?;
    #   ...
    #   let last_hidden = hidden_states.squeeze(0)?.get(hidden_states.dim(1)? - 1)?;
    #   generated_hidden_states.push(last_hidden);
    #   ...
    # }
    # So it collects the hidden state of the PRESENTLY input token.
    # The first input is the prompt: [STOP][TEXT]hello world![START]
    # The last token in prompt is [START].
    # So generated_hidden_states[0] is the HS of [START].
    
    # Let's check Python's extraction in soprano/backends/transformers.py
    # and soprano/tts.py

if __name__ == "__main__":
    debug_inference()

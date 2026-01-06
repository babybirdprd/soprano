import json

try:
    with open("tokenizer.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if "model" in data and "ignore_merges" in data["model"]:
        print("Removing ignore_merges from model...")
        del data["model"]["ignore_merges"]
        
        with open("tokenizer.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("Done.")
    else:
        print("ignore_merges not found in model.")

except Exception as e:
    print(f"Error: {e}")

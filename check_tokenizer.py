from tokenizers import Tokenizer

try:
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file("tokenizer.json")
    print("Successfully loaded tokenizer!")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
except Exception as e:
    print(f"Failed to load tokenizer: {e}")

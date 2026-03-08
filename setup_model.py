"""
One-time setup script: Downloads the base Llama-3 model to the local HuggingFace cache.

Run this ONCE with internet access:
    python setup_model.py

After this runs successfully, the app (streamlit run app.py) will work
100% offline using local_files_only=True.
"""

import os
import sys

from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not found in .env file.")
    print('Create a .env file with: HF_TOKEN=hf_your_token_here')
    sys.exit(1)

from huggingface_hub import login
login(token=HF_TOKEN)

print("=" * 60)
print("Downloading base model to local HuggingFace cache...")
print("This only needs to happen ONCE. After this, everything is offline.")
print("=" * 60)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

print(f"\n[1/2] Downloading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Tokenizer cached successfully.\n")

print(f"[2/2] Downloading model weights for {MODEL_NAME}...")
print("(This is ~5GB and may take several minutes)\n")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)

print("\n" + "=" * 60)
print("SUCCESS: Base model is now cached locally.")
print(f"Cache location: {os.path.expanduser('~/.cache/huggingface/hub/')}")
print("")
print("You can now run the app fully offline:")
print("  streamlit run app.py")
print("=" * 60)

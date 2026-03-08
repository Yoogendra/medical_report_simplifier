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

try:
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

except Exception as e:
    error_msg = str(e)
    if "403 Client Error" in error_msg or "gated repo" in error_msg.lower():
        print("\n" + "❌ " * 20)
        print("ERROR: YOU DO NOT HAVE ACCESS TO THIS MODEL YET.")
        print("Llama-3 is a 'gated' model. You must accept Meta's license first.")
        print("\nFix this in 3 easy steps:")
        print("1. Go to: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct")
        print("2. Log in with the account that matches your HF_TOKEN")
        print("3. Fill out the form and click 'Agree and access repository'")
        print("\nOnce granted (usually instant), run this script again.")
        print("❌ " * 20 + "\n")
    else:
        print(f"\nAn unexpected error occurred: {e}")
    sys.exit(1)

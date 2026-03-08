"""
Llama-3 Inference Engine — Offline-First, Privacy-Preserving.

Loading strategy:
1. Authenticate via HF_TOKEN from .env (needed for gated model access to HF cache)
2. Load base model + PEFT adapter from local files only (local_files_only=True)
3. Lock down environment to offline mode after loading
"""

import os
import sys
import yaml
import torch
import logging
from pathlib import Path

# ──────────────────────────────────────────────────
# STEP 1: Security — Load HF_TOKEN from .env
# ──────────────────────────────────────────────────
from dotenv import load_dotenv

load_dotenv()  # loads .env from project root

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("=" * 60)
    print("ERROR: HF_TOKEN not found.")
    print("Create a .env file in the project root with:")
    print('  HF_TOKEN=hf_your_token_here')
    print("You can get a token from https://huggingface.co/settings/tokens")
    print("=" * 60)
    sys.exit(1)

# Authenticate with HuggingFace (writes token to local cache for gated model access)
from huggingface_hub import login
login(token=HF_TOKEN, add_to_git_credential=False)

# ──────────────────────────────────────────────────
# STEP 2: Force Offline Mode — set BEFORE any model loading
# ──────────────────────────────────────────────────
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["WANDB_MODE"] = "disabled"

logger = logging.getLogger(__name__)


def get_device():
    """Auto-detects the best available hardware: CUDA > Apple MPS > CPU."""
    if torch.cuda.is_available():
        logger.info("Device: NVIDIA CUDA GPU detected.")
        return "cuda"
    elif torch.backends.mps.is_available():
        logger.info("Device: Apple Silicon MPS (Metal) detected.")
        return "mps"
    else:
        logger.info("Device: No GPU detected, falling back to CPU.")
        return "cpu"


class LlamaInference:
    def __init__(self, config_path="config/model_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model_config = self.config["model"]
        self.gen_config = self.config["generation"]
        self.device = get_device()

        self._validate_paths()
        self._load_model()

    def _validate_paths(self):
        """Path validation — fail fast with a clear message."""
        adapter_path = self.model_config["adapter_path"]
        if not Path(adapter_path).exists():
            print("=" * 60)
            print("Error: PEFT Adapter not found.")
            print(f"Please place your files in {adapter_path}/")
            print("Expected files: adapter_config.json, adapter_model.safetensors,")
            print("                tokenizer.json, tokenizer_config.json")
            print("=" * 60)
            sys.exit(1)

        adapter_config = Path(adapter_path) / "adapter_config.json"
        if not adapter_config.exists():
            print("=" * 60)
            print(f"Error: adapter_config.json not found in {adapter_path}/")
            print("This doesn't look like a valid PEFT adapter directory.")
            print("=" * 60)
            sys.exit(1)

        logger.info(f"Path validation passed: {adapter_path}")

    def _load_model(self):
        """
        Two-stage offline loading:
        1. Load base Llama-3 model from local HF cache (local_files_only=True)
        2. Apply the local PEFT LoRA adapter on top

        Memory optimization:
          - CUDA : 4-bit BitsAndBytes quantization (65% memory reduction)
          - MPS  : fp16 via Apple Metal + device_map="auto"
          - CPU  : fp16 + device_map="auto"
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        adapter_path = self.model_config["adapter_path"]
        base_model_name = self.model_config["base_model"]

        # ── Load tokenizer from LOCAL adapter folder ──
        logger.info(f"Loading tokenizer from local adapter: {adapter_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_path,
            local_files_only=True,
        )

        # ── Load base model from LOCAL HF cache ──
        logger.info(f"Loading base model from local cache: {base_model_name}")

        if self.device == "cuda" and self.model_config.get("load_in_4bit", True):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            logger.info("4-bit BitsAndBytes quantization active (65% memory reduction).")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=quantization_config,
                device_map="auto",
                local_files_only=True,
            )
        else:
            # MPS (Apple Silicon) and CPU — use fp16 + device_map auto
            logger.info(f"Loading base model in fp16 on {self.device}.")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                local_files_only=True,
            )

        # ── Apply PEFT LoRA adapter ──
        logger.info(f"Applying PEFT LoRA adapter from: {adapter_path}")
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        logger.info("Model loaded successfully. Running in offline mode.")

    def generate(self, prompt: str) -> str:
        """Generates a response using the locally loaded fine-tuned Llama-3."""
        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.gen_config["max_new_tokens"],
                use_cache=self.gen_config["use_cache"],
                temperature=self.gen_config.get("temperature", 0.1),
                top_p=self.gen_config.get("top_p", 0.9),
                do_sample=True,
            )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

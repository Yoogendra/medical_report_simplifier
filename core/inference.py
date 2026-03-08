import yaml
import torch
import logging
import os
from pathlib import Path
from utils.privacy_engine import PrivacyEngine

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

        self._load_model()

        # Enforce offline mode AFTER model is loaded into memory
        # (base model may need one-time download to HF cache on first run)
        PrivacyEngine.enforce_offline_mode()

    def _load_model(self):
        """
        Two-stage loading:
        1. Load base Llama-3 model from HF cache (or download once if not cached)
        2. Apply the local PEFT LoRA adapter on top

        Quantization:
          - CUDA : 4-bit BitsAndBytes (65% memory reduction)
          - MPS  : fp16 via Apple Metal
          - CPU  : bf16
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        adapter_path = self.model_config["adapter_path"]
        base_model_name = self.model_config["base_model"]

        if not Path(adapter_path).exists():
            raise FileNotFoundError(
                f"Adapter folder not found at '{adapter_path}'.\n"
                "Download your trained adapter from Google Drive and place its contents there."
            )

        # Load tokenizer from the local adapter folder (it was saved with the adapter)
        logger.info(f"Loading tokenizer from local adapter: {adapter_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_path,
            local_files_only=False,  # tokenizer files are local
        )

        # Load base model — from HF cache if already downloaded, else fetches once
        logger.info(f"Loading base model: {base_model_name}")
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
            )
        elif self.device == "mps":
            logger.info("Loading base model in fp16 for Apple Silicon MPS.")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
            ).to(self.device)
        else:
            logger.info("Loading base model in bf16 on CPU.")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
            )

        # Apply the PEFT LoRA adapter on top of the base model
        logger.info(f"Applying PEFT LoRA adapter from: {adapter_path}")
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        logger.info("Model ready.")

    def generate(self, prompt: str) -> str:
        """Generates a response using locally loaded fine-tuned Llama-3."""
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

import yaml
import torch
import logging
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
        # Enforce offline mode on import
        PrivacyEngine.enforce_offline_mode()

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model_config = self.config["model"]
        self.gen_config = self.config["generation"]
        self.device = get_device()

        self._load_model()

    def _load_model(self):
        """
        Loads the fine-tuned model saved by the notebook's model.save_pretrained().
        The notebook used Unsloth's save which produces a full merged causal LM
        (not a separate PEFT adapter), so we load it as a standard CausalLM.

        Quantization strategy:
          - CUDA  : 4-bit BitsAndBytes (65% memory reduction)
          - MPS   : fp16 via Apple Metal
          - CPU   : bf16 fallback
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        model_path = self.model_config["adapter_path"]

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found at '{model_path}'.\n"
                "Please download your trained adapter from Google Drive and place it there.\n"
                "Expected files: adapter_config.json / config.json, *.safetensors, tokenizer files."
            )

        logger.info(f"Loading tokenizer from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self.device == "cuda" and self.model_config.get("load_in_4bit", True):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            logger.info("Loading model with 4-bit BitsAndBytes quantization (65% memory reduction).")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
            )
        elif self.device == "mps":
            logger.info("Loading model in fp16 for Apple Silicon MPS.")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
            ).to(self.device)
        else:
            logger.info("Loading model in bf16 on CPU.")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            )

        self.model.eval()
        logger.info("Model loaded successfully and set to eval mode.")

    def generate(self, prompt: str) -> str:
        """Generates a response using the locally loaded fine-tuned model."""
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

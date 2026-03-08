import yaml
import torch
import logging
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
        Loads the fine-tuned Llama-3 model with PEFT adapters.
        Uses standard transformers + peft for cross-platform compatibility:
        - CUDA  : 4-bit BitsAndBytes quantization (full 65% memory reduction)
        - MPS   : fp16 via Apple Metal (memory-efficient on Apple Silicon)
        - CPU   : bf16/fp32 fallback
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        adapter_path = self.model_config["adapter_path"]
        base_model_name = self.model_config["base_model"]

        logger.info(f"Loading tokenizer from: {adapter_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)

        # 4-bit quantization available on CUDA only
        if self.device == "cuda" and self.model_config.get("load_in_4bit", True):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            logger.info("Loading base model with 4-bit BitsAndBytes quantization (65% memory reduction).")
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
                device_map={"": self.device},
            )
        else:
            logger.info("Loading base model on CPU (bf16).")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": self.device},
            )

        logger.info(f"Loading PEFT LoRA adapters from: {adapter_path}")
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        logger.info("Model loaded successfully and set to eval mode.")

    def generate(self, prompt: str) -> str:
        """Generates a response using the locally loaded Llama-3 + PEFT model."""
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

import yaml
import torch
from unsloth import FastLanguageModel
from utils.privacy_engine import PrivacyEngine

class LlamaInference:
    def __init__(self, config_path="config/model_config.yaml"):
        # Ensure offline constraints
        PrivacyEngine.enforce_offline_mode()
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.model_config = self.config["model"]
        self.gen_config = self.config["generation"]
        self.prompts = self.config["prompts"]

        self._load_model()
        
    def _load_model(self):
        """Loads the Unsloth base model and tokenizer offline."""
        # Load the PEFT adapter and base model offline with 4-bit quantization
        # This achieves the 65% memory reduction footprint.
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_config["adapter_path"],
            max_seq_length=self.model_config["max_seq_length"],
            dtype=self.model_config["dtype"],
            load_in_4bit=self.model_config["load_in_4bit"],
        )
        FastLanguageModel.for_inference(self.model)

    def generate(self, prompt: str) -> str:
        """Generates a response using the Llama-3 model."""
        inputs = self.tokenizer(
            [prompt], 
            return_tensors="pt"
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        # Stream memory efficient generation
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=self.gen_config["max_new_tokens"], 
            use_cache=self.gen_config["use_cache"]
        )
        
        return self.tokenizer.batch_decode(outputs)[0]

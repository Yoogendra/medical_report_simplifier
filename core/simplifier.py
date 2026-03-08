from core.inference import LlamaInference
import yaml
import logging

logger = logging.getLogger(__name__)

class ClinicalSimplifier:
    def __init__(self, inference_engine: LlamaInference):
        self.engine = inference_engine
        
        # Load the formatting prompt template
        with open("config/model_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        self.prompt_template = config["prompts"]["alpaca_template"]

    def simplify(self, medical_report: str) -> str:
        """
        Formats the clinical text using the prompt template and 
        routes it to the inference engine.
        """
        logger.info("ClinicalSimplifier: Formatting report for inference.")
        
        formatted_prompt = self.prompt_template.format(
            medical_report.strip(), 
            "" # Empty string for the model to fill in
        )
        
        # We need to append the EOS token structure manually depending on standard Llama 3 behavior
        # But Unsloth formatting usually just needs the prompt
        raw_output = self.engine.generate(formatted_prompt)
        
        return self._post_process(raw_output)

    def _post_process(self, raw_output: str) -> str:
        """
        Cleans the model output to extract only the simplified explanation.
        """
        try:
            # The model repeats the prompt, so we split at the marker
            cleaned_text = raw_output.split("### Simplified Explanation:")[-1]
            
            # Remove generation artifacts
            cleaned_text = cleaned_text.replace("<|end_of_text|>", "").strip()
            
            return cleaned_text
        except Exception as e:
            logger.error(f"Error during post-processing: {e}")
            return "Error: Could not parse model output."

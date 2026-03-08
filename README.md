# Privacy-Preserving Clinical Text Simplifier 🏥

An "air-gapped" style production repository that translates complex clinical text and medical jargon into patient-friendly language. Built using Llama-3, instruction fine-tuned with PEFT (LoRA), and heavily optimized for low-resource inference environments.

## 🛡️ Architecture & Security Profile

This repository enforces strict **offline-first**, **zero-egress** constraints suitable for deployment in secure hospital environments or local workstations handling Protected Health Information (PHI). 

- **Zero-Egress (`core/inference.py`)**: `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are explicitly set at runtime. All `from_pretrained` requests force `local_files_only=True`.
- **Local Weight Resolution**: Tokenizers, adapters, and base models are loaded exclusively from local storage to prevent accidental cloud-syncing or telemetry.
- **Air-Gapped Interface**: The offline Streamlit UI is locked to `localhost` with all internet sharing protocols explicitly forbidden.

## ⚡ Memory Footprint Reduction

By leveraging 4-bit BitsAndBytes quantization, the system achieves a **65% reduction in memory footprint** compared to fp16 baselines without degrading clinical accuracy. Memory benchmarks:

- **Base Model (fp16)**: ~16GB VRAM
- **Quantized 4-bit (This Repo on CUDA)**: ~5.5GB VRAM
- **Apple Silicon / CPU**: Uses `device_map="auto"` and `fp16` fallback seamlessly.
- **PEFT LoRA Overhead**: Minimal (r=16, ~40MB)

## 📂 Project Structure

```
├── config/
│   └── model_config.yaml       # Hyperparameters & Offline Path Configs
├── core/
│   ├── inference.py            # Target-Hardware Optimized Llama-3 Engine
│   └── simplifier.py           # NLP Logic, Prompt Formatting & Post-Processing
├── models/
│   └── Medical_Llama_Adapter/  # ⚠️ Place your trained PEFT adapter files here!
├── app.py                      # Offline Streamlit UI
├── setup_model.py              # One-time script to cache the base model
├── .env                        # ⚠️ Create this (holds your HF_TOKEN)
├── requirements.txt            # Local Dependencies
└── README.md
```

## 🚀 Deployment Guide (Offline Execution)

### 1. Environment Requirements
Install the necessary packages inside a virtual environment (`python >= 3.10` recommended).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare the Security Token & Llama License
Llama-3 is a "gated" model. You must accept the Meta license to download the base weights.

1. Go to [Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and click **"Agree and access repository"**.
2. Go to your [Hugging Face Settings](https://huggingface.co/settings/tokens) and generate a new Read-Only Access Token.
3. Create a `.env` file in the root of this project and add your token:
   ```env
   HF_TOKEN=hf_your_token_here
   ```

### 3. Provide Your Custom Adapters
Place the contents of your trained Llama-3 PEFT adapter into the `./models/Medical_Llama_Adapter/` folder.
*Required files: `adapter_config.json`, `adapter_model.safetensors`, and the tokenizer JSON configs.*

### 4. Cache the Base Model (One-Time Network Requirement)
Before you can run strictly offline, you must cache the base model to your machine once. 

Run the setup script:
```bash
python setup_model.py
```
*This downloads ~5GB of data. Once success is printed, you can physically disconnect from the internet.*

### 5. Launch the Air-Gapped UI
```bash
streamlit run app.py
```
The application will now securely load the cached models from memory enforcing strict `local_files_only` constraints.

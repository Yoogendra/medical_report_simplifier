# Privacy-Preserving Clinical Text Simplifier 🏥

An "air-gapped" style production repository that translates complex clinical text and medical jargon into patient-friendly language. Built using Llama-3, instruction fine-tuned with PEFT (LoRA), and heavily optimized for low-resource inference environments.

## 🛡️ Architecture & Security Profile

This repository enforces strict **offline-first**, **zero-egress** constraints suitable for deployment in secure hospital environments or local workstations handling Protected Health Information (PHI). 

- **Zero-Egress (`utils/privacy_engine.py`)**: Hugging Face Hub telemetry is disabled. Online connection attempts from dependent libraries are blocked by explicit environment overrides.
- **Local Weight Resolution**: Tokenizers, adapters, and base models are loaded exclusively from the local (`./models/`) directory to prevent accidental cloud-syncing.
- **Air-Gapped Interface**: The Gradio/Streamlit UI is locked to `localhost` with all internet sharing protocols (`localtunnel`, `share=True`) explicitly forbidden.

## ⚡ Memory Footprint Reduction

By leveraging the Unsloth framework and BitsAndBytes for 4-bit quantization, the system achieves a **65% reduction in memory footprint** compared to fp16 baselines without degrading clinical accuracy. Memory benchmarks:

- **Base Model (fp16)**: ~16GB VRAM
- **Quantized 4-bit (This Repo)**: ~5.5GB VRAM
- **PEFT LoRA Overhead**: Minimal (r=16, ~40MB)

## 📂 Project Structure

```
├── config/
│   └── model_config.yaml       # Hyperparameters & Offline Path Configs
├── core/
│   ├── inference.py            # Llama-3 4-bit Quantized Engine
│   └── simplifier.py           # NLP Logic, Prompt Formatting & Post-Processing
├── models/
│   └── Medical_Llama_Adapter/  # (Ignored) Your trained PEFT adapters go here
├── utils/
│   └── privacy_engine.py       # Zero-Egress Enforcement Engine
├── app.py                      # Offline Streamlit UI
├── requirements.txt            # Local Dependencies
└── README.md
```

## 🚀 Quickstart (Offline Deployment)

### 1. Environment Setup

Install the required packages in an isolated virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

*Note: Unsloth must be installed separately depending on your CUDA architecture:*
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### 2. Supply Local Model Weights

Place your instruction fine-tuned Llama-3 PEFT adapters in the designated local directory. The default path defined in `config/model_config.yaml` is:
`./models/Medical_Llama_Adapter/`

*(Never commit these weights to version control!)*

### 3. Run the Medical Simplifier

Launch the Local Application:

```bash
streamlit run app.py
```

The system will initialize the `PrivacyEngine` to enforce offline mode, load the 4-bit quantized base model and adapters, and serve the application locally at `http://localhost:8501`.

import streamlit as st
import logging

from core.inference import LlamaInference
from core.simplifier import ClinicalSimplifier

logging.basicConfig(level=logging.INFO)

# 1. Page Configuration
st.set_page_config(
    page_title="Medical Report Simplifier",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Medical Report Simplifier")
st.markdown("### Turn complex doctor notes into plain English.")
st.caption("🔒 Runs completely offline on your local machine. No internet required.")

# 2. Load Model (cached so it only loads once)
@st.cache_resource
def load_system():
    engine = LlamaInference()
    simplifier = ClinicalSimplifier(engine)
    return simplifier

with st.spinner("Loading AI model locally... (this may take a minute the first time)"):
    try:
        simplifier = load_system()
        st.success("✅ Model loaded and running locally.")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.info("Make sure your fine-tuned adapter files are placed in the `./models/Medical_Llama_Adapter/` folder.")
        st.stop()

# 3. User Interface
input_text = st.text_area(
    "Paste Medical Report Here:",
    height=200,
    placeholder="Patient presents with acute pharyngitis and cervical lymphadenopathy..."
)

if st.button("Simplify Report ✨"):
    if input_text:
        with st.spinner("Generating simplified explanation..."):
            simplified_text = simplifier.simplify(input_text)
            st.success("Here is the simplified version:")
            st.write(simplified_text)
    else:
        st.warning("Please enter some text first!")

st.markdown("---")
st.caption("Powered by Llama-3 + PEFT | Privacy-First, Offline Operation | Zero Egress")

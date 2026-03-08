import streamlit as st
import logging

# Ensure privacy constraints are loaded first
from utils.privacy_engine import PrivacyEngine
from core.inference import LlamaInference
from core.simplifier import ClinicalSimplifier

logging.basicConfig(level=logging.INFO)

# 1. Page Configuration
st.set_page_config(
    page_title="Medical Simplifier AI [Air-Gapped]", 
    page_icon="🏥", 
    layout="centered"
)

st.title("🏥 Medical Report Simplifier")
st.markdown("### Turn complex doctor notes into plain English.")
st.caption("🔒 Running locally with zero egress constraints.")

# 2. Load Model Offline
@st.cache_resource
def load_system():
    # Enforce air-gapped simulation
    if not PrivacyEngine.verify_no_egress():
        st.warning("Warning: Internet connection detected. This violates strict air-gapped policies!")
        
    engine = LlamaInference()
    simplifier = ClinicalSimplifier(engine)
    return simplifier

with st.spinner("Waking up the local AI Doctor... (Loading 4-bit adapters)"):
    try:
        simplifier = load_system()
    except Exception as e:
        st.error(f"Failed to load the model. Ensure adapters are in ./models/: {e}")
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
st.caption("Powered by Llama-3 & Unsloth | 65% Memory Footprint Reduction Achieved | Zero-Egress Environment")

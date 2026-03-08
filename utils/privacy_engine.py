import os
import socket
import logging

logger = logging.getLogger(__name__)

class PrivacyEngine:
    """
    Enforces privacy-first, zero-egress constraints for air-gapped environments.
    """
    
    @staticmethod
    def enforce_offline_mode():
        """
        Sets required environment variables to prevent Hugging Face and other
        libraries from attempting to connect to the internet.
        """
        # Disable HuggingFace Hub telemetry and online connection attempts
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
        # Disable Weights & Biases if installed
        os.environ["WANDB_OFFLINE"] = "true"
        os.environ["WANDB_MODE"] = "dryrun"

        logger.info("PrivacyEngine: Offline mode enforced. Zero-egress constraints applied.")

    @staticmethod
    def verify_no_egress():
        """
        A simple check to ensure no external internet connection can be made,
        simulating an air-gapped environment. In a true air-gapped environment,
        this should raise an exception or timeout.
        """
        try:
            # Try to connect to a common public DNS
            socket.create_connection(("8.8.8.8", 53), timeout=1)
            logger.warning("PrivacyEngine WARNING: External internet connection detected! "
                           "This violates air-gapped constraints.")
            return False
        except OSError:
            logger.info("PrivacyEngine: Verified no internet egress (Air-gapped confirmed).")
            return True

# Initialize offline mode aggressively upon import
PrivacyEngine.enforce_offline_mode()

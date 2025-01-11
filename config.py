import os
import torch
from typing import List

def determine_batch_size(device: str) -> int:
    """
    Determine the batch size dynamically based on the available device and resources.

    :param device: The computation device ("cuda" or "cpu").
    :return: Recommended batch size for the device.
    """
    if device == "cuda":
        # Check GPU properties
        import torch
        gpu_properties = torch.cuda.get_device_properties(0)
        vram_gb = gpu_properties.total_memory / (1024 ** 3)  # Convert VRAM to GB

        # Set batch size based on VRAM
        if vram_gb >= 16:  # High-end GPU
            return 256
        elif vram_gb >= 8:  # Mid-range GPU
            return 128
        else:  # Low-end GPU
            return 64
    else:
        # CPU-based batch size (depends on core count)
        core_count = os.cpu_count()
        if core_count >= 16:
            return 32
        elif core_count >= 8:
            return 16
        else:
            return 8

CONFIG = {
    "embedding_file": "context_embeddings.pkl",
    "faiss_index_file": "faiss_index.bin",
    "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "qa_model_name": "deepset/roberta-base-squad2",
    "squad_split": "train[:50%]",
    "DEFAULT_TOP_K_CONTEXTS": 5,
    "K_VALUES_FOR_METRICS": [1, 3, 5],
    "BATCH_SIZE": None,  # To be set based on DEVICE
    "NUM_PROC": max(os.cpu_count() - 2, 1),
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
}

# Initialize BATCH_SIZE based on DEVICE
CONFIG["BATCH_SIZE"] = determine_batch_size(CONFIG["DEVICE"])
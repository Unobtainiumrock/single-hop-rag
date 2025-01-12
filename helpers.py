import numpy as np
import pickle
import faiss
import torch
import os

from loggers import main_logger  # Import the logger
from typing import Tuple, List
from config import CONFIG

DEVICE = CONFIG.get("DEVICE")

# ------------------- Save/Load Embeddings and Index -------------------


def save_embeddings(data: Tuple[np.ndarray, List[str]], file_path: str) -> None:
    """
    Save embeddings and contexts to a file.
    
    :param data: A tuple containing the numpy array of embeddings and a list of contexts.
    :param file_path: The path to the file where embeddings and contexts will be stored.
    """
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    main_logger.info(f"Embeddings and contexts saved to {file_path}.")


def load_embeddings(file_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load embeddings and contexts from a file.

    :param file_path: The path to the file where embeddings and contexts are stored.
    :return: A tuple containing the numpy array of embeddings and a list of contexts.
    """
    with open(file_path, "rb") as f:
        embeddings, contexts = pickle.load(f)
    main_logger.info(f"Embeddings and contexts loaded from {file_path}.")
    return embeddings, contexts


def save_faiss_index(index: faiss.Index, file_path: str) -> None:
    """
    Save a FAISS index to a file, ensuring it is on the CPU.

    :param index: The FAISS index to save.
    :param file_path: The path to the file where the index will be saved.
    :return: None
    """
    # Ensure the index is on the CPU before saving
    if DEVICE == "cuda" and hasattr(index, "index_gpu_to_cpu"):
        index = faiss.index_gpu_to_cpu(index)

    faiss.write_index(index, file_path)
    main_logger.info(f"FAISS index saved to {file_path}.")


def load_faiss_index(file_path: str) -> faiss.Index:
    """
    Load a FAISS index from a file.

    :param file_path: The path to the file where the index is stored.
    :return: The loaded FAISS index.
    """
    index = faiss.read_index(file_path)
    main_logger.info(f"FAISS index loaded from {file_path}.")
    return index


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

def transfer_faiss_to_gpu():
    """
    Transfer the FAISS index to GPU.
    """
    main_logger.info("Transferring FAISS index to GPU...")
    try:
        res = faiss.StandardGpuResources()
        resources["faiss_index"] = faiss.index_cpu_to_gpu(res, 0, resources["faiss_index"])
        main_logger.info("FAISS index successfully transferred to GPU.")
    except RuntimeError as e:
        main_logger.warning(f"GPU transfer failed. Using CPU index: {str(e)}")
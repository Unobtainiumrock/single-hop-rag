import numpy as np
import pickle
import faiss
import torch
import os

from loggers import main_logger  # Import the logger

DEVICE = "cuda" if torch.cuda.is_available() else ""

# ------------------- Save/Load Embeddings and Index -------------------


def save_embeddings(embeddings: np.ndarray, file_path: str) -> None:
    """
    Save embeddings to a file.

    :param embeddings: The numpy array of embeddings to save.
    :param file_path: The path to the file where embeddings will be saved.
    :return: None
    """
    with open(file_path, "wb") as f:
        pickle.dump(embeddings, f)
    main_logger.info(f"Embeddings saved to {file_path}.")


def load_embeddings(file_path: str) -> np.ndarray:
    """
    Load embeddings from a file.

    :param file_path: The path to the file where embeddings are stored.
    :return: The loaded numpy array of embeddings.
    """
    with open(file_path, "rb") as f:
        embeddings = pickle.load(f)
    main_logger.info(f"Embeddings loaded from {file_path}.")
    return embeddings


def save_faiss_index(index: faiss.Index, file_path: str) -> None:
    """
    Save a FAISS index to a file (CPU index only).

    :param index: The FAISS index to save.
    :param file_path: The path to the file where the index will be saved.
    :return: None
    """
    if isinstance(index, faiss.IndexFlatL2) and DEVICE == "cuda":
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

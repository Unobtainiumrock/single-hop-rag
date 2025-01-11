import numpy as np
import pickle
import faiss
import torch

from loggers import main_logger  # Import the logger

DEVICE = "cuda" if torch.cuda.is_available() else ""

# ------------------- Save/Load Embeddings and Index -------------------


def save_embeddings(embeddings: np.ndarray, file_path: str) -> None:
    """Saves embeddings to a file."""
    with open(file_path, "wb") as f:
        pickle.dump(embeddings, f)
    main_logger.info(f"Embeddings saved to {file_path}.")


def load_embeddings(file_path: str) -> np.ndarray:
    """Loads embeddings from a file."""
    with open(file_path, "rb") as f:
        embeddings = pickle.load(f)
    main_logger.info(f"Embeddings loaded from {file_path}.")
    return embeddings


def save_faiss_index(index: faiss.Index, file_path: str) -> None:
    """Saves a FAISS index to a file (CPU index only)."""
    if isinstance(index, faiss.IndexFlatL2) and DEVICE == "cuda":
        index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, file_path)
    main_logger.info(f"FAISS index saved to {file_path}.")


def load_faiss_index(file_path: str) -> faiss.Index:
    """Loads a FAISS index from a file."""
    index = faiss.read_index(file_path)
    main_logger.info(f"FAISS index loaded from {file_path}.")
    return index

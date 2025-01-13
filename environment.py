import os
import faiss
# import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
# import faiss
from datasets import load_dataset, Dataset
from loggers import main_logger

from helpers import (
    load_embeddings,
    save_embeddings,
    load_faiss_index,
    save_faiss_index,
    # determine_batch_size
)

from embeddings import generate_embeddings_with_batching
from config import CONFIG
from resources import resources

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CONFIG = {
#     "embedding_file": "context_embeddings.pkl",
#     "faiss_index_file": "faiss_index.bin",
#     "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
#     "qa_model_name": "deepset/roberta-base-squad2",
#     "squad_split": "train[:50%]",
#     "DEFAULT_TOP_K_CONTEXTS": 5,
#     "K_VALUES_FOR_METRICS": [1, 3, 5],  # Precision@K, Recall@K
#     "BATCH_SIZE": determine_batch_size(DEVICE),
#     "NUM_PROC": max(os.cpu_count - 2, 1)
# }

# resources: Dict[str, any] = {
#     "embedding_model": None,
#     "qa_pipeline": None,
#     "faiss_index": None,
#     "contexts": []
# }


def initialize_environment() -> None:
    """
    Initialize the environment by setting up models, embeddings, and FAISS index.

    This function prepares the embedding model, QA pipeline, and FAISS index. If pre-existing embeddings
    and index files are found, they are loaded. Otherwise, embeddings are generated, normalized,
    and saved, and a new FAISS index is created. The FAISS index is transferred to GPU if available.

    :raises RuntimeError: If model or FAISS initialization fails.
    """
    mandatory_config_keys = [
        "embedding_file",
        "faiss_index_file",
        "embedding_model_name",
        "qa_model_name",
        "squad_split",
        "DEFAULT_TOP_K_CONTEXTS",
        "K_VALUES_FOR_METRICS",
        "BATCH_SIZE",
        "NUM_PROC"
    ]

    missing_keys = [key for key in mandatory_config_keys if key not in CONFIG]

    if missing_keys:
        main_logger.error(f"Missing configuration parameters: {missing_keys}")
        raise ValueError(f"Missing configuration parameters: {missing_keys}")

    embedding_file = CONFIG.get("embedding_file")
    faiss_index_file = CONFIG.get("faiss_index_file")
    embedding_model_name = CONFIG.get("embedding_model_name")
    qa_model_name = CONFIG.get("qa_model_name")
    squad_split = CONFIG.get("squad_split")

    DEVICE = CONFIG.get("DEVICE")

    # Initialize Embedding Model
    main_logger.info("Initializing embedding model...")
    try:
        resources["embedding_model"] = SentenceTransformer(
            embedding_model_name, device=DEVICE)
        main_logger.info(f"Embedding model loaded on device: {DEVICE}")
    except Exception as e:
        main_logger.error(f"Failed to initialize embedding model: {str(e)}")
        raise

       # Initialize QA Pipeline
    main_logger.info("Initializing QA pipeline...")

    try:
        resources["qa_pipeline"] = pipeline(
            "question-answering", model=qa_model_name, device=0 if DEVICE == "cuda" else -1)
        main_logger.info(f"QA model loaded: {qa_model_name}")
    except Exception as e:
        main_logger.error(f"Failed to initialize QA pipeline: {str(e)}")
        raise

    # Load or Generate Embeddings and FAISS Index
    if os.path.exists(embedding_file) and os.path.exists(faiss_index_file):
        main_logger.info("Loading existing embeddings and FAISS index...")
        context_embeddings, contexts = load_embeddings(embedding_file)
        resources["faiss_index"] = load_faiss_index(faiss_index_file)
        resources["contexts"] = contexts
    else:
        main_logger.info(
            "No existing data found. Generating embeddings and creating FAISS index...")

        # Load dataset
        dataset = load_dataset("squad", split=squad_split)

        # Extract unique contexts
        unique_contexts = list(np.unique(dataset["context"]))
        resources["contexts"] = unique_contexts
        main_logger.info(f"Extracted {len(unique_contexts)} unique contexts.")

        unique_contexts_dataset = Dataset.from_dict(
            {"context": unique_contexts})

        # Generate embeddings with batching
        context_embeddings = generate_embeddings_with_batching(
            unique_contexts_dataset)
        main_logger.info(f"Generated embeddings for unique contexts.")

        save_embeddings((context_embeddings, unique_contexts), embedding_file)

        # Create FAISS index
        resources["faiss_index"] = faiss.IndexFlatL2(
            context_embeddings.shape[1])
        resources["faiss_index"].add(context_embeddings)
        save_faiss_index(resources.get("faiss_index"), faiss_index_file)

    # When transferring the FAISS index to the GPU, the embeddings already stored in the FAISS index are automatically moved to the GPU as part of the index.
    if DEVICE == "cuda":
        try:
            main_logger.info("Transferring FAISS index to GPU for querying...")
            res = faiss.StandardGpuResources()
            resources["faiss_index"] = faiss.index_cpu_to_gpu(
                res, 0, resources.get("faiss_index"))
            main_logger.info("FAISS index successfully transferred to GPU.")
        except RuntimeError as e:
            main_logger.warning(
                f"GPU transfer failed. Using CPU index: {str(e)}")

import numpy as np
from typing import List, Dict, Callable, Any
# from environment import resources, DEVICE, CONFIG
from loggers import main_logger, embed_logger
from datasets import Dataset
from config import CONFIG
from resources import resources


def normalize_embeddings(func: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
    """
    Decorator to normalize embeddings returned by the wrapped function.

    Each row of the output embeddings will be normalized to have a unit L2 norm.

    :param func: The function that returns embeddings to be normalized.
    :return: A wrapped function that normalizes the embeddings.
    :raises ValueError: If the output of the wrapped function is not a numpy array.
    """

    def wrapper(*args: Any, **kwargs: Any) -> np.ndarray:
        embeddings = func(*args, **kwargs)
        if not isinstance(embeddings, np.ndarray):
            raise ValueError("Embedding output must be a numpy array.")

        # Normalize each row/vector
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # To avoid division by zero for any zero vector
        norms[norms == 0] = 1e-12
        return (embeddings / norms).astype(np.float32)

    return wrapper


@normalize_embeddings
def encode_query(query: str) -> np.ndarray:
    """
    Encode a single query into a normalized embedding.

    :param query: The query string to encode.
    :return: A 2D array of shape `(1, dim)` containing the normalized embedding.
    :raises ValueError: If the query is not a non-empty string.
    """
    embedding_model = resources.get("embedding_model")

    # Validate input
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    # Encode the query
    q_embedding = embedding_model.encode(query, convert_to_tensor=False)

    # Ensure the result is a 2D numpy array with a single row
    return np.asarray([q_embedding], dtype=np.float32)


@normalize_embeddings
def encode_texts(texts: List[str]) -> np.ndarray:
    """
    Encode a batch of texts into normalized float32 embeddings.

    This function uses the global `embedding_model` to encode input texts into numerical embeddings.

    :param texts: A list of strings to encode.
    :return: An array of shape `(N, dim)` containing the normalized embeddings.
    """
    embed_logger.info("Encoding texts into embeddings...")
    embedding_model = resources.get("embedding_model")
    embeddings = embedding_model.encode(texts, convert_to_tensor=False)
    return np.asarray(embeddings, dtype=np.float32)


def embed_contexts(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Embed a batch of contexts and append normalized embeddings to the batch.

    :param batch: A dictionary with a "context" key containing a list of strings to embed.
    :return: The input batch with an additional "embeddings" key containing the embeddings.
    """
    embed_logger.info(f"Embedding batch of size {len(batch['context'])}.")

    # Encode and normalize embeddings
    emb = encode_texts(batch.get("context"))
    batch["embeddings"] = emb

    embed_logger.info("Batch embedding completed.")
    return batch


# @normalize_embeddings
# def encode_query(query: str) -> np.ndarray:
#     """
#     Encode a single query into a normalized embedding.

#     :param query: The query string to encode.
#     :return: A 2D array of shape `(1, dim)` containing the normalized embedding.
#     :raises ValueError: If the query is not a non-empty string.
#     """

#     # Validate input
#     if not isinstance(query, str) or not query.strip():
#         raise ValueError("Query must be a non-empty string.")

#     # Encode the query
#     q_embedding = embedding_model.encode(query, convert_to_tensor=False)

#     # Ensure the result is a 2D numpy array with a single row
#     return np.asarray([q_embedding], dtype=np.float32)


def generate_embeddings_with_batching(unique_contexts_dataset: Dataset) -> np.ndarray:
    """
    Generate embeddings for unique contexts using batching with the dataset.map function.

    :param unique_contexts_dataset: A Hugging Face Dataset containing only unique contexts.
    :return: A numpy array of embeddings.
    """
    main_logger.info(
        "Generating embeddings with batching using dataset.map...")
    embedding_model = resources.get("embedding_model")
    if embedding_model is None:
        main_logger.error("Embedding model is not initialized in resources.")
        raise ValueError("Embedding model not found in resources.")

    # Define the embedding function for dataset.map
    def embed_contexts(batch):
        contexts = batch["context"]
        embeddings = embedding_model.encode(
            contexts,
            convert_to_tensor=False,
            show_progress_bar=False,
            # Use CONFIG batch size or default to 64
            batch_size=CONFIG.get("BATCH_SIZE", 64)
        )
        return {"embeddings": np.asarray(embeddings, dtype=np.float32)}

    # Use dataset.map for batched processing
    embedded_dataset = unique_contexts_dataset.map(
        embed_contexts,
        batched=True,
        batch_size=CONFIG.get("BATCH_SIZE", 64),
        num_proc=1 if CONFIG.get(
            "DEVICE", "cpu") == "cuda" else CONFIG.get("NUM_PROC", 4)
    )

    # Stack all embeddings into a single numpy array
    context_embeddings = np.vstack(
        embedded_dataset["embeddings"]).astype(np.float32)
    main_logger.info(f"Generated embeddings shape: {context_embeddings.shape}")
    return context_embeddings

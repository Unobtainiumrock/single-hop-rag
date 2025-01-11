from typing import Dict, Any
import logging
import os
import json
from typing import List, Dict, Callable, Any

import numpy as np
import faiss
import torch

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from flask import Flask, request, jsonify

from loggers import main_logger, embed_logger
from compute_config_helpers import determine_batch_size
from helpers import save_embeddings, load_embeddings, save_faiss_index, load_faiss_index

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --------------------------- Configuration ----------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOG_FILE = "app.log"
EMBEDDING_FILE = "context_embeddings.pkl"
FAISS_INDEX_FILE = "faiss_index.bin"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL_NAME = "deepset/roberta-base-squad2"
SQUAD_SPLIT = "train[:50%]"

DEFAULT_TOP_K_CONTEXTS = 5
K_VALUES_FOR_METRICS = [1, 3, 5]  # Precision@K, Recall@K

dataset = load_dataset("squad", split=SQUAD_SPLIT)

# Globals
embedding_model = None
qa_pipeline = None
CONTEXTS: List[str] = dataset["context"]
INDEX: faiss.IndexFlatL2 = None

# Batch Processing Config
BATCH_SIZE = determine_batch_size(DEVICE)
NUM_PROC = max(os.cpu_count() - 2, 1)  # Leave 2 CPUs free

# ------------------------ Environment Initialization ------------------------


def initialize_environment() -> None:
    """Initialize the environment by setting up models, embeddings, and FAISS index."""
    global INDEX, CONTEXTS, NUM_PROC, embedding_model, qa_pipeline, dataset

    # Initialize Embedding Model
    main_logger.info("Initializing embedding model...")
    try:
        embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME, device=DEVICE)
        main_logger.info(f"Embedding model loaded on device: {DEVICE}")
    except Exception as e:
        main_logger.error(f"Failed to initialize embedding model: {str(e)}")
        raise

    # Initialize QA Pipeline
    main_logger.info("Initializing QA pipeline...")
    try:
        qa_pipeline = pipeline(
            "question-answering", model=QA_MODEL_NAME, device=0 if DEVICE == "cuda" else -1
        )
        main_logger.info(f"QA model loaded: {QA_MODEL_NAME}")
    except Exception as e:
        main_logger.error(f"Failed to initialize QA pipeline: {str(e)}")
        raise

    # Load or Generate Embeddings and FAISS Index
    if os.path.exists(EMBEDDING_FILE) and os.path.exists(FAISS_INDEX_FILE):
        main_logger.info("Loading existing embeddings and FAISS index...")
        context_embeddings = load_embeddings(EMBEDDING_FILE)
        INDEX = load_faiss_index(FAISS_INDEX_FILE)
    else:
        main_logger.info(
            "No existing data found. Generating embeddings and creating FAISS index...")

        # Generate embeddings
        dataset = dataset.map(
            embed_contexts,
            batched=True,
            batch_size=BATCH_SIZE,
            num_proc=1 if DEVICE == "cuda" else NUM_PROC
        )
        context_embeddings = np.vstack(dataset["embeddings"])

    # Ensure embeddings are on CPU and of correct type before saving to disk
    if torch.is_tensor(context_embeddings):
        context_embeddings = context_embeddings.cpu().numpy()
    if context_embeddings.dtype != np.float32:
        context_embeddings = context_embeddings.astype(np.float32)

    save_embeddings(context_embeddings, EMBEDDING_FILE)

    # Create FAISS index
    INDEX = faiss.IndexFlatL2(context_embeddings.shape[1])
    INDEX.add(context_embeddings)
    save_faiss_index(INDEX, FAISS_INDEX_FILE)

    # When transferring the FAISS index to the GPU, the embeddings already stored in the FAISS index are automatically moved to the GPU as part of the index.
    if DEVICE == "cuda":
        try:
            main_logger.info("Transferring FAISS index to GPU for querying...")
            res = faiss.StandardGpuResources()
            INDEX = faiss.index_cpu_to_gpu(res, 0, INDEX)
            main_logger.info("FAISS index successfully transferred to GPU.")
        except RuntimeError as e:
            main_logger.warning(
                f"GPU transfer failed. Using CPU index: {str(e)}")


# ------------------------ Helper Functions ------------------------


def normalize_embeddings(func: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
    """
    A decorator to normalize embeddings returned by the wrapped function.

    The wrapped function must return a numpy array of shape `(N, dim)`,
    where `N` is the number of embeddings and `dim` is the embedding dimension.
    Each row (embedding) in the array will be normalized to have unit L2 norm.

    Raises:
        ValueError: If the output of the wrapped function is not a numpy array.

    Args:
        func (Callable[..., np.ndarray]): The function to be decorated. It should return a numpy array.

    Returns:
        Callable[..., np.ndarray]: A wrapper function that normalizes the embeddings returned by `func`.
    """
    def wrapper(*args: Any, **kwargs: Any) -> np.ndarray:
        embeddings = func(*args, **kwargs)
        if not isinstance(embeddings, np.ndarray):
            raise ValueError("Embedding output must be a numpy array.")

        # Normalize each row/vector
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # To avoid division by zero for any zero vector
        norms[norms == 0] = 1e-12
        return embeddings / norms

    return wrapper


@normalize_embeddings
def encode_texts(texts: List[str]) -> np.ndarray:
    """
    Encodes a batch of texts into float32 embeddings and normalizes them.

    This function uses the `embedding_model` to convert a list of input texts
    into numerical embeddings. The embeddings are returned as a numpy array
    of type `float32` and are normalized to have unit L2 norm for each row.

    Args:
        texts (List[str]): A list of input strings to be encoded.

    Returns:
        np.ndarray: A numpy array of shape `(N, dim)` where `N` is the number
        of input texts and `dim` is the dimensionality of the embeddings. Each
        row in the array represents the normalized embedding of a corresponding text.
    """
    emb = embedding_model.encode(texts, convert_to_tensor=False)
    return np.asarray(emb, dtype=np.float32)


def embed_contexts(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Embeds a batch of contexts and appends the resulting normalized embeddings to the batch.

    This function takes a dictionary representing a batch of data, where `batch["context"]`
    contains a list of textual contexts. It encodes these contexts into numerical embeddings,
    normalizes the embeddings, and appends them to the batch under the key `batch["embeddings"]`.

    Args:
        batch (Dict[str, Any]): A dictionary containing the key "context", which is expected
        to be a list of strings representing the textual data to embed.

    Returns:
        Dict[str, Any]: The input batch dictionary with an additional key, "embeddings",
        which contains a numpy array of shape `(N, dim)` where `N` is the number of contexts
        and `dim` is the embedding dimensionality. Each row in the array represents the
        normalized embedding of the corresponding context.

    Raises:
        KeyError: If the "context" key is missing from the input batch.
        ValueError: If `batch["context"]` is not a list of strings.
    """
    logger = logging.getLogger("embed_contexts")

    # Validate input batch
    if "context" not in batch:
        raise KeyError("The input batch must contain a 'context' key.")
    if not isinstance(batch["context"], list) or not all(isinstance(ctx, str) for ctx in batch["context"]):
        raise ValueError("The 'context' key must be a list of strings.")

    logger.info(f"Embedding batch of size {len(batch['context'])}.")

    # Encode and normalize embeddings
    emb = encode_texts(batch["context"])
    batch["embeddings"] = emb

    logger.info("Batch embedding completed.")
    return batch

# -------------------- Retrieval & QA Functions --------------------


@normalize_embeddings
def encode_query(query: str) -> np.ndarray:
    """
    Encodes a single query into a normalized embedding.

    This function uses the `embedding_model` to encode a single input query string 
    into a numerical embedding. The embedding is returned as a normalized numpy 
    array of shape `(1, dim)`, where `dim` is the dimensionality of the embedding.

    Args:
        query (str): The input query string to be encoded.

    Returns:
        np.ndarray: A numpy array of shape `(1, dim)` containing the normalized embedding.

    Raises:
        ValueError: If the input query is not a string or is empty.
    """
    # Validate input
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    # Encode the query
    q_embedding = embedding_model.encode(query, convert_to_tensor=False)

    # Ensure the result is a 2D numpy array with a single row
    return np.asarray([q_embedding], dtype=np.float32)


def retrieve_contexts(query: str,
                      top_k: int = DEFAULT_TOP_K_CONTEXTS) -> List[str]:
    """
    Retrieve the top_k most relevant contexts for a given query using the global FAISS index.

    This function encodes the input query into a normalized embedding, performs a search 
    on the FAISS index, and retrieves the corresponding contexts from the global `CONTEXTS` list.

    Args:
        query (str): The input query string for which relevant contexts are to be retrieved.
        top_k (int): The number of top contexts to retrieve. Defaults to `DEFAULT_TOP_K_CONTEXTS`.

    Returns:
        List[str]: A list of up to `top_k` retrieved contexts, ranked by relevance.

    Raises:
        ValueError: If the global `CONTEXTS` or `INDEX` is not initialized, if `top_k` is not a 
        positive integer, or if `top_k` exceeds the number of available contexts.
        RuntimeError: If the FAISS search fails due to an unexpected error.
    """
    # Validate global variables
    if not CONTEXTS or INDEX is None:
        raise ValueError("CONTEXTS or INDEX is not initialized.")

    # Validate top_k
    if top_k <= 0:
        raise ValueError("`top_k` must be a positive integer.")
    if top_k > len(CONTEXTS):
        raise ValueError(
            "`top_k` cannot exceed the number of available contexts.")

    main_logger.debug(
        "Retrieving contexts for query: '%s', top_k: %d", query, top_k)

    # Encode the query (normalized by the decorator)
    query_embedding = encode_query(query)

    # Perform FAISS search
    try:
        # Look into the _ = distances later.
        _, indices = INDEX.search(query_embedding, top_k)
    except Exception as e:
        main_logger.error(f"FAISS search failed: {str(e)}")
        raise RuntimeError("Error during FAISS search.") from e

    # Handle case where no results are returned
    if indices.size == 0:
        main_logger.warning("No contexts retrieved for query: '%s'", query)
        return []

    # Retrieve the actual contexts based on indices
    retrieved_contexts = [CONTEXTS[idx] for idx in indices[0]]
    main_logger.debug("Retrieved %d contexts for query.",
                      len(retrieved_contexts))

    return retrieved_contexts


def generate_answer(query: str,
                    top_k: int = DEFAULT_TOP_K_CONTEXTS
                    ) -> Dict[str, Any]:
    """
    Generates an answer to a query by retrieving relevant contexts and using a QA pipeline.

    This function retrieves the top_k most relevant contexts for the given query using 
    the `retrieve_contexts` function, then uses a QA pipeline to generate answers 
    based on those contexts. It returns the best answer along with all retrieved answers.

    Args:
        query (str): The input query string for which an answer is to be generated.
        top_k (int): The number of top contexts to retrieve. Defaults to `DEFAULT_TOP_K_CONTEXTS`.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - "best_answer" (str): The best answer generated by the QA pipeline, or None if no answer is found.
            - "all_answers" (List[Dict[str, Any]]): A list of all answers with their scores, sorted in descending order.

    Raises:
        ValueError: If `retrieve_contexts` fails due to uninitialized globals or invalid parameters.
        RuntimeError: If the QA pipeline encounters an unexpected error.
    """
    main_logger.debug("Generating answer for query: %s", query)

    # Retrieve relevant contexts
    try:
        relevant_contexts = retrieve_contexts(query, top_k=top_k)
    except ValueError as e:
        main_logger.error(f"Error retrieving contexts: {str(e)}")
        return {
            "best_answer": None,
            "all_answers": []
        }

    # Batch the question-context pairs for faster processing
    try:
        raw_answers = qa_pipeline([
            {"question": query, "context": cxt} for cxt in relevant_contexts
        ])
    except Exception as e:
        main_logger.error(f"QA pipeline error: {str(e)}")
        raw_answers = []

    # Sort answers by score in descending order
    answers_sorted = sorted(
        raw_answers, key=lambda x: x["score"], reverse=True) if raw_answers else []
    best_answer = answers_sorted[0] if answers_sorted else {
        "answer": None, "score": 0.0}

    return {
        "best_answer": best_answer["answer"],
        "all_answers": answers_sorted
    }

# ----------------- Evaluation Metrics Functions ------------------


def evaluate_rag(questions: List[str],
                 ground_truth_answers: List[Any],
                 k_values: List[int],
                 top_k_contexts: int = DEFAULT_TOP_K_CONTEXTS
                 ) -> Dict[str, Any]:
    """
    Evaluates the RAG (Retrieval-Augmented Generation) system by calculating precision@k and recall@k.

    This function computes the precision and recall metrics at various values of `k` for a set of
    questions. It retrieves the top_k most relevant contexts for each question, generates answers 
    using the QA pipeline, and compares the generated answers against the ground truth.

    Args:
        questions (List[str]): A list of input questions to evaluate.
        ground_truth_answers (List[Any]): The ground truth answers corresponding to each question.
            Each element can be a single answer (str) or a list of answers (List[str]).
        k_values (List[int]): A list of integer values for `k` to calculate precision@k and recall@k.
        top_k_contexts (int): The number of contexts to retrieve for each query. Defaults to `DEFAULT_TOP_K_CONTEXTS`.

    Returns:
        Dict[str, Any]: A dictionary containing precision@k and recall@k for each value in `k_values`.

    Raises:
        ValueError: If `questions` or `ground_truth_answers` are empty, or if their lengths do not match.
    """
    main_logger.info("Starting RAG evaluation...")

    # Validate input
    if not questions:
        raise ValueError("No questions provided for evaluation.")
    if not ground_truth_answers:
        raise ValueError("No ground truth answers provided for evaluation.")
    if len(questions) != len(ground_truth_answers):
        raise ValueError(
            "Mismatch between the number of questions and ground truth answers.")

    # Ensure ground_truth_answers is a list of lists
    if not all(isinstance(gt, list) for gt in ground_truth_answers):
        ground_truth_answers = [
            [gt] if isinstance(gt, str) else gt
            for gt in ground_truth_answers
        ]

    # Initialize metrics
    metrics = {f"precision@{k}": 0.0 for k in k_values}
    metrics.update({f"recall@{k}": 0.0 for k in k_values})
    total_questions = len(questions)

    # Iterate over each question-answer pair
    for i, (question, ground_truth) in enumerate(zip(questions, ground_truth_answers)):
        main_logger.info(
            f"Evaluating question {i + 1}/{total_questions}: {question}")

        # Retrieve top_k contexts
        try:
            retrieved_contexts = retrieve_contexts(
                question, top_k=top_k_contexts)
        except Exception as e:
            main_logger.error(
                f"Error retrieving contexts for question '{question}': {str(e)}")
            continue

        # Generate answers using the QA pipeline
        try:
            raw_answers = [
                qa_pipeline(question=question, context=cxt)
                for cxt in retrieved_contexts
            ]
            raw_answers_sorted = sorted(
                raw_answers, key=lambda x: x["score"], reverse=True
            )
        except Exception as e:
            main_logger.error(
                f"Error generating answers for question '{question}': {str(e)}")
            continue

        # Extract the top-k answers
        retrieved_answers = [
            ans["answer"] for ans in raw_answers_sorted[:max(k_values)]
        ]

        # Calculate metrics for each k
        for k in k_values:
            top_k_answers = retrieved_answers[:k]

            # Precision@k: Proportion of retrieved answers that match the ground truth
            precision = sum(
                1 for ans in top_k_answers if ans in ground_truth) / k
            metrics[f"precision@{k}"] += precision

            # Recall@k: Proportion of ground truth answers covered by top-k retrieved answers
            recall = sum(
                1 for ans in top_k_answers if ans in ground_truth) / len(ground_truth)
            metrics[f"recall@{k}"] += recall

    # Normalize metrics by the total number of questions
    for k in k_values:
        metrics[f"precision@{k}"] /= total_questions
        metrics[f"recall@{k}"] /= total_questions

    main_logger.info("RAG evaluation completed.")
    return metrics


# -------------------------- Flask App -----------------------------
app = Flask(__name__)


@app.route("/ask", methods=["POST"])
def ask_question():
    """
    Handles /ask endpoint:
    - Accepts a JSON payload with a "question" and optional "top_k_contexts".
    - Returns the best answer and all retrieved answers.
    """
    main_logger.info("Received request at /ask")
    data = request.json

    # Input validation
    question = data.get("question")
    if not question:
        main_logger.error("No question provided")
        return jsonify({"error": "No question provided", "code": 400}), 400

    # Retrieve top_k (optional)
    top_k_contexts = data.get("top_k_contexts", DEFAULT_TOP_K_CONTEXTS)

    # Generate answer
    try:
        gen_dict = generate_answer(question, top_k=top_k_contexts)
    except Exception as e:
        main_logger.error("Error generating answer: %s", str(e))
        return jsonify({"error": "Internal server error", "code": 500}), 500

    # Prepare response
    response = {
        "question": question,
        "best_answer": gen_dict["best_answer"],
        "all_answers": gen_dict["all_answers"],
    }

    # Log success
    main_logger.info("Successfully processed question: '%s', Best Answer: '%s'",
                     question, gen_dict["best_answer"])

    # Return response (optionally prettified if requested)
    pretty = data.get("pretty", False)  # Optionally prettify via JSON
    if pretty:
        return json.dumps(response, indent=4), 200, {'Content-Type': 'application/json'}
    else:
        return jsonify(response), 200


@app.route("/evaluate", methods=["POST"])
def evaluate_model():
    """
    Handles /evaluate endpoint:
    - Accepts a JSON payload with "questions" and "ground_truth_answers".
    - Optionally accepts "top_k_contexts" and "k_values".
    - Returns evaluation metrics (precision@k, recall@k).
    """
    main_logger.info("Received request at /evaluate")
    data = request.json

    # Input validation
    questions = data.get("questions")
    ground_truth_answers = data.get("ground_truth_answers")
    if not questions or not ground_truth_answers:
        main_logger.error("Questions or ground truth answers not provided")
        return jsonify({"error": "Questions or ground truth answers not provided", "code": 400}), 400

    # Optional parameters
    top_k_contexts = data.get("top_k_contexts", DEFAULT_TOP_K_CONTEXTS)
    k_values = data.get("k_values", K_VALUES_FOR_METRICS)

    # Perform evaluation
    try:
        metrics = evaluate_rag(
            questions, ground_truth_answers, k_values=k_values, top_k_contexts=top_k_contexts
        )
    except Exception as e:
        main_logger.error("Error during evaluation: %s", str(e))
        return jsonify({"error": "Internal server error", "code": 500}), 500

    # Log success
    main_logger.info("Evaluation completed successfully: Metrics=%s", metrics)

    # Return response
    pretty = data.get("pretty", False)  # Optionally prettify via JSON
    if pretty:
        return json.dumps(metrics, indent=4), 200, {'Content-Type': 'application/json'}
    else:
        return jsonify(metrics), 200


def main():
    """Main entry point for the application."""
    initialize_environment()
    main_logger.info(
        "Environment initialization complete. Starting Flask app...")
    # app.run(port=5000)


# -------------------------- Quick Tests ---------------------------
if __name__ == "__main__":

    main()

    # Hardcoded test context
    test_context = """
    France is a country in Western Europe. The capital of France is Paris,
    which is known for its art, fashion, gastronomy and culture.
    """
    answer = qa_pipeline(
        question="What is the capital of France?", context=test_context
    )
    print("Rough test on hardcoded context:", answer)

    # Example queries
    test_queries = [
        "What is the capital of France?",
        "Who wrote Hamlet?",
        "What is photosynthesis?"
    ]
    for i, query in enumerate(test_queries):
        main_logger.info("\n=== Testing Query: %s ===", query)
        retrieved_contexts = retrieve_contexts(
            query, top_k=DEFAULT_TOP_K_CONTEXTS)
        print(f"\nQuery: {query}")
        for j, cxt in enumerate(retrieved_contexts):
            print(f"Context {j + 1}:\n{cxt[:300]}...\n")

        print(
            f"Answer {i + 1}:\n{json.dumps(generate_answer(query), indent=2)}\n")
    # app.run(port=5000)

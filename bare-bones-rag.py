import logging
import os
import json
import pickle

import numpy as np
import faiss

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from flask import Flask, request, jsonify
from typing import List, Dict, Any, Tuple, Set

# ------------------------ Compute Config ------------------------
total_cpus = os.cpu_count()
usable_cpus = total_cpus - 2  # Adjust as needed

# ------------------------ Logging Config ------------------------
def setup_logger(name: str, level=logging.INFO, log_file: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler (optional)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

# Main logger
main_logger = setup_logger("main_logger", level=logging.INFO, log_file="app.log")

# Logger for embedding contexts (child processes)
embed_logger = setup_logger("embed_contexts", level=logging.WARNING)

# Suppress verbosity from external libraries
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING) 

# ---------------------- Global/Default Config --------------------
DEFAULT_TOP_K_CONTEXTS = 5
K_VALUES_FOR_METRICS   = [1, 3, 5]  # Precision@K, Recall@K

# ------------------------ Model + Files --------------------------
main_logger.info("Instantiating the embedding model...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedding_file = "context_embeddings.pkl"
faiss_index_file = "faiss_index.bin"

# Global variables
contexts: List[str] = []
index: faiss.IndexFlatL2 = None

# ------------------- Normalization Decorator --------------------
def normalize_embeddings(func):
    """
    Decorator to normalize embeddings returned by the wrapped function.
    Expects a numpy array of shape (N, dim).
    """
    def wrapper(*args, **kwargs):
        embeddings = func(*args, **kwargs)
        if isinstance(embeddings, np.ndarray):
            # Normalize each row/vector
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # To avoid division by zero for any zero vector
            norms[norms == 0] = 1e-12
            return embeddings / norms
        else:
            raise ValueError("Embedding output must be a numpy array.")
    return wrapper

# ------------------- Utility: Single-Batch Embed --------------------
@normalize_embeddings
def encode_texts(texts: List[str]) -> np.ndarray:
    """
    Encodes a batch of texts into float32 embeddings and normalizes them.
    """
    emb = embedding_model.encode(texts, convert_to_tensor=False)
    emb = np.asarray(emb, dtype=np.float32)
    return emb

# ------------------- Utility: Map-based Embed --------------------
def embed_contexts(batch):
    """
    Takes a batch of contexts (batch["context"]) and returns a batch with
    a new key "embeddings" that contains normalized float32 embeddings.
    """
    logger = logging.getLogger("embed_contexts")
    logger.info(f"Embedding batch of size {len(batch['context'])}.")
    
    # 1) Encode the batch
    emb = encode_texts(batch["context"])  # The decorator normalizes them
    
    # 2) Assign back to the batch
    batch["embeddings"] = emb
    
    logger.info("Batch embedding completed.")
    return batch

# ------------------- Building / Loading Index --------------------
if os.path.exists(embedding_file) and os.path.exists(faiss_index_file):
    main_logger.info("Loading existing embeddings and FAISS index...")
    
    # Load embeddings
    with open(embedding_file, "rb") as f:
        context_embeddings: np.ndarray = pickle.load(f)
    
    main_logger.info("Loaded context_embeddings: shape=%s, dtype=%s",
                    context_embeddings.shape, context_embeddings.dtype)
    
    # Double-check they're float32
    if context_embeddings.dtype != np.float32:
        context_embeddings = context_embeddings.astype(np.float32)
        main_logger.info("Converted context_embeddings to float32.")
    
    # Normalize again just to be sure. 
    # No harm as vector normalization is invariant under subsequent normalization
    norms = np.linalg.norm(context_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    context_embeddings = context_embeddings / norms
    
    # Load FAISS index
    index = faiss.read_index(faiss_index_file)
    main_logger.info("FAISS index loaded from %s", faiss_index_file)
    
    # Load dataset to populate `contexts`
    dataset = load_dataset("squad", split="train[:50%]")
    contexts = dat  aset["context"]  # Store as a simple list
    contexts, questions, answers = dataset["context"], dataset["question"], dataset["answers"]
    print("=" * 50 + "\n")
    print(f"Question:{questions[0]}" + "\n")
    print(f"Answer: {answers[0]}" + "\n")
    print(f"Context:{contexts[0]}" + "\n    ")
    print("=" * 50)

else:
    main_logger.info("No existing data found; generating new embeddings and FAISS index...")
    
    # Load dataset
    dataset = load_dataset("squad", split="train[:50%]")
    contexts = dataset["context"]  # Simple list
    main_logger.info("Dataset loaded. Number of contexts: %d", len(contexts))
    
    # Embed all contexts with map
    main_logger.info("Batch embedding contexts with HF map + normalization...")
    dataset = dataset.map(
        embed_contexts,
        batched=True,
        batch_size=64,
        num_proc=usable_cpus
    )
    
    # Convert list-of-lists embeddings into single array
    context_embeddings = np.vstack(dataset["embeddings"])
    main_logger.info("Combined embeddings: shape=%s, dtype=%s",
                     context_embeddings.shape, context_embeddings.dtype)
    
    # Save embeddings
    with open(embedding_file, "wb") as f:
        pickle.dump(context_embeddings, f)
    main_logger.info("Embeddings saved to %s", embedding_file)
    
    # Create FAISS index
    dimension = context_embeddings.shape[1]
    main_logger.info("Creating FAISS index (dimension=%d)...", dimension)
    index = faiss.IndexFlatL2(dimension)
    index.add(context_embeddings)
    main_logger.info("FAISS index created. Total vectors in index: %d", index.ntotal)
    
    # Save FAISS index
    faiss.write_index(index, faiss_index_file)
    main_logger.info("FAISS index saved to %s", faiss_index_file)

# ---------------------- Retrieval & QA Logic ----------------------
@normalize_embeddings
def encode_query(query: str) -> np.ndarray:
    """
    Encodes a single query into an array of shape (1, dim), normalizes it.
    """
    q_embedding = embedding_model.encode(query, convert_to_tensor=False)
    return np.asarray([q_embedding], dtype=np.float32)

def retrieve_contexts(query: str, top_k: int = DEFAULT_TOP_K_CONTEXTS) -> List[str]:
    """
    Retrieve top_k contexts from the global `contexts` using the FAISS index.
    """
    main_logger.debug("Retrieving context for query: %s", query)
    
    # 1) Encode query (normalized by the decorator)
    query_embedding = encode_query(query)
    
    # 2) FAISS search
    distances, indices = index.search(query_embedding, top_k)
    main_logger.debug("Distances shape=%s, Indices shape=%s", distances.shape, indices.shape)
    
    # 3) Retrieve actual contexts
    retrieved_contexts = [contexts[idx] for idx in indices[0]]
    return retrieved_contexts

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def generate_answer(query: str, top_k: int = DEFAULT_TOP_K_CONTEXTS) -> Dict[str, Any]:
    """
    Retrieve top_k contexts for the query, then use QA pipeline to get an answer from each context.
    Returns a dict with:
      - best_answer
      - all_answers
    """
    main_logger.debug("Generating answer for query: %s", query)
    
    relevant_contexts = retrieve_contexts(query, top_k=top_k)
    
    raw_answers = [qa_pipeline(question=query, context=cxt)
                   for cxt in relevant_contexts]
    # Sort by score desc
    answers_sorted = sorted(raw_answers, key=lambda x: x["score"], reverse=True)
    best_answer = answers_sorted[0]
    
    return {
        "best_answer": best_answer["answer"],
        "all_answers": answers_sorted
    }

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
    pretty = data.get("pretty", False)  # Optionally prettify via query param
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
        metrics = evaluate_rag(questions, ground_truth_answers,
                               k_values=k_values,
                               top_k_contexts=top_k_contexts)
    except Exception as e:
        main_logger.error("Error during evaluation: %s", str(e))
        return jsonify({"error": "Internal server error", "code": 500}), 500

    # Log success
    main_logger.info("Evaluation completed successfully: Metrics=%s", metrics)

    # Return response
    pretty = data.get("pretty", False)  # Optionally prettify via query param
    if pretty:
        return json.dumps(metrics, indent=4), 200, {'Content-Type': 'application/json'}
    else:
        return jsonify(metrics), 200

    # -------------------------- Quick Tests ---------------------------
if __name__ == "__main__":
    # Hardcoded test context
    test_context = """
    France is a country in Western Europe. The capital of France is Paris,
    which is known for its art, fashion, gastronomy and culture.
    """
    answer = qa_pipeline(question="What is the capital of France?", context=test_context)
    print("Rough test on hardcoded context:", answer)
    
    # Example queries
    test_queries = [
        "What is the capital of France?",
        "Who wrote Hamlet?",
        "What is photosynthesis?"
    ]
    for i, query in enumerate(test_queries):
        main_logger.info("\n=== Testing Query: %s ===", query)
        retrieved_contexts = retrieve_contexts(query, top_k=DEFAULT_TOP_K_CONTEXTS)
        print(f"\nQuery: {query}")
        for j, cxt in enumerate(retrieved_contexts):
            print(f"Context {j + 1}:\n{cxt[:300]}...\n")
    
        print(f"Answer {i + 1}:\n{json.dumps(generate_answer(query), indent=2)}\n")

    # if __name__ == "__main__":
    #     app.run(port=5000)

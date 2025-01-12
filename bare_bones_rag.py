import faiss
import numpy as np
import os
import json

from typing import List, Dict, Any
from flask import Flask, request, jsonify

from loggers import main_logger
from environment import resources, CONFIG, initialize_environment
from embeddings import encode_query

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# -------------------- Retrieval & QA Functions --------------------


def retrieve_contexts(query: str, top_k: int = CONFIG.get("DEFAULT_TOP_K_CONTEXTS")) -> List[str]:
    """
    Retrieve the top_k most relevant contexts for a query using the FAISS index.

    :param query: The input query string.
    :param top_k: The number of top contexts to retrieve. Defaults to `DEFAULT_TOP_K_CONTEXTS`.
    :return: A list of retrieved contexts ranked by relevance.
    :raises ValueError: If `contexts` or `faiss_index` is not initialized, or if `top_k` is invalid.
    :raises RuntimeError: If FAISS search fails.
    """
    # Validate global variables
    faiss_index = resources.get("faiss_index")
    contexts = resources.get("contexts")

    if not contexts or faiss_index is None:
        raise ValueError("Contexts or FAISS index is not initialized.")

    if top_k <= 0:
        raise ValueError("`top_k` must be a positive integer.")
    if top_k > len(contexts):
        raise ValueError("`top_k` cannot exceed the number of available contexts.")

    main_logger.debug("Retrieving contexts for query: '%s', top_k: %d", query, top_k)

    # Encode the query (normalized by the decorator)
    query_embedding = encode_query(query)

    # Perform FAISS search
    try:
        _, indices = faiss_index.search(query_embedding, top_k)
    except Exception as e:
        main_logger.error(f"FAISS search failed: {str(e)}")
        raise RuntimeError("Error during FAISS search.") from e

    # Handle case where no results are returned
    if indices.size == 0:
        main_logger.warning("No contexts retrieved for query: '%s'", query)
        return []

    # Retrieve the actual contexts based on indices
    retrieved_contexts = [contexts[idx] for idx in indices[0]]
    main_logger.debug("Retrieved %d contexts for query.", len(retrieved_contexts))

    return retrieved_contexts


def generate_answer(query: str, top_k: int = CONFIG.get("DEFAULT_TOP_K_CONTEXTS")) -> Dict[str, Any]:
    """
    Generate an answer to a query using relevant contexts and a QA pipeline.

    :param query: The input query string.
    :param top_k: The number of top contexts to retrieve. Defaults to `DEFAULT_TOP_K_CONTEXTS`.
    :return: A dictionary containing the best answer and all retrieved answers.
    :raises ValueError: If context retrieval fails.
    :raises RuntimeError: If the QA pipeline encounters an error.
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
        qa_pipeline = resources.get("qa_pipeline")
        batch_inputs = [{"question": query, "context": cxt} for cxt in relevant_contexts]
        raw_answers = qa_pipeline(batch_inputs)
    except Exception as e:
        main_logger.error(f"QA pipeline error: {str(e)}")
        raw_answers = []

    # Sort answers by score in descending order
    answers_sorted = sorted(raw_answers, key=lambda x: x["score"], reverse=True) if raw_answers else []
    best_answer = answers_sorted[0] if answers_sorted else {
        "answer": None, "score": 0.0}

    return {
        "best_answer": best_answer["answer"],
        "all_answers": answers_sorted
    }

# ----------------- Evaluation Metrics Functions ------------------


def evaluate_rag(
    questions: List[str],
    ground_truth_answers: List[Any],
    k_values: List[int],
    top_k_contexts: int = CONFIG.get("DEFAULT_TOP_K_CONTEXTS")
) -> Dict[str, float]:
    """
    Evaluate the Retrieval-Augmented Generation (RAG) system using precision@k and recall@k.

    :param questions: List of questions to evaluate. Shape: [num_questions]
    :param ground_truth_answers: Ground truth answers for each question.
                                 Shape: [num_questions] or [num_questions, num_answers]
    :param k_values: List of k values for computing precision and recall metrics.
                    Example: [1, 5, 10]
    :param top_k_contexts: Number of contexts to retrieve for each query. Defaults to CONFIG["DEFAULT_TOP_K_CONTEXTS"].
    :return: Dictionary containing precision@k and recall@k metrics.
             Keys are in the format "precision@k" and "recall@k".
    :raises ValueError: If inputs are invalid or mismatched.
    """
    main_logger.info("Starting RAG evaluation...")

    # Validate input lengths
    if not questions:
        raise ValueError("No questions provided for evaluation.")
    if not ground_truth_answers:
        raise ValueError("No ground truth answers provided for evaluation.")
    if len(questions) != len(ground_truth_answers):
        raise ValueError("Mismatch between the number of questions and ground truth answers.")

    # Ensure ground_truth_answers is a list of sets for efficient membership checks
    ground_truth_answers = [
        set(gt) if isinstance(gt, list) else {gt}
        for gt in ground_truth_answers
    ]

    # Initialize metrics dictionary
    metrics: Dict[str, float] = {f"precision@{k}": 0.0 for k in k_values}
    metrics.update({f"recall@{k}": 0.0 for k in k_values})
    total_questions = len(questions)

    # Precompute the maximum k value
    max_k = max(k_values)

    # Iterate over each question and its corresponding ground truth answers
    for i, (question, ground_truth) in enumerate(zip(questions, ground_truth_answers)):
        main_logger.info(f"Evaluating question {i + 1}/{total_questions}: {question}")

        # Retrieve top_k_contexts for the current question
        try:
            retrieved_contexts: List[str] = retrieve_contexts(question, top_k=top_k_contexts)
            # retrieved_contexts shape: [top_k_contexts]
        except Exception as e:
            main_logger.error(f"Error retrieving contexts for question '{question}': {str(e)}")
            continue

        # Generate answers using the QA pipeline
        try:
            # Prepare batch inputs: list of dicts with 'question' and 'context'
            batch_inputs: List[Dict[str, str]] = [{"question": question, "context": cxt} for cxt in retrieved_contexts]
            raw_answers: List[Dict[str, Any]] = qa_pipeline(batch_inputs)
            # raw_answers shape: [top_k_contexts], each dict contains 'answer' and 'score'

            # Sort answers by their confidence score in descending order
            raw_answers_sorted: List[Dict[str, Any]] = sorted(
                raw_answers, key=lambda x: x["score"], reverse=True
            )
        except Exception as e:
            main_logger.error(f"Error generating answers for question '{question}': {str(e)}")
            continue

        # Extract the top-k answers up to the maximum k value
        retrieved_answers: List[str] = [
            ans["answer"] for ans in raw_answers_sorted[:max_k]
        ]
        # retrieved_answers shape: [max_k]

        # Calculate metrics for each k (k_values = [1, 3, 5] by default)
        for k in k_values:
            # Ensure k does not exceed the number of retrieved answers
            current_k = min(k, len(retrieved_answers))
            if current_k == 0:
                continue  # Avoid division by zero

            # Top-k answers
            top_k_answers = retrieved_answers[:current_k]
            # top_k_answers shape: [current_k]

            # Precision@k: Proportion of retrieved answers that are correct
            correct = sum(ans in ground_truth for ans in top_k_answers)
            precision = correct / current_k
            metrics[f"precision@{k}"] += precision

            # Recall@k: Proportion of ground truth answers that are retrieved
            recall = correct / len(ground_truth)
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
    top_k_contexts = data.get("top_k_contexts", CONFIG.get("DEFAULT_TOP_K_CONTEXTS"))

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
    top_k_contexts = data.get("top_k_contexts", CONFIG.get("DEFAULT_TOP_K_CONTEXTS"))
    k_values = data.get("k_values", CONFIG.get("K_VALUES_FOR_METRICS"))

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

    # Initialize the environment: load models, embeddings, FAISS index
    initialize_environment()
    main_logger.info("Environment initialization complete.")


    # Hardcoded test context
    test_context = """
    France is a country in Western Europe. The capital of France is Paris,
    which is known for its art, fashion, gastronomy and culture.
    """
    qa_pipeline = resources.get("qa_pipeline")

    answer = qa_pipeline(
        question="What is the capital of France?", context=test_context
    )
    print("Rough test on hardcoded context:", answer)

    # (Optional) Run some example queries to test RAG functions
    test_queries = [
        "What is the capital of Russia?",
        "Who wrote Hamlet?",
        "What is photosynthesis?"
    ]

    for i, query in enumerate(test_queries):
        main_logger.info("\n=== Testing Query: %s ===", query)
        retrieved_contexts = retrieve_contexts(
            query, top_k=CONFIG.get("DEFAULT_TOP_K_CONTEXTS"))
        print(f"\nQuery: {query}")
        for j, cxt in enumerate(retrieved_contexts):
            print(f"Context {j + 1}:\n{cxt[:300]}...\n")

        answer = generate_answer(query)
        print(f"Answer {i + 1}:\n{json.dumps(answer, indent=2)}\n")

    # Start Flask app (to be moved to app.py later)
    main_logger.info("Starting Flask app...")
    app.run(port=5000)

if __name__ == "__main__":
    main()


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Calculate Precision@K.
    :param retrieved: List of retrieved contexts.
    :param relevant: List of ground truth relevant contexts.
    :param k: Number of top contexts to consider.
    :return: Precision@K value.
    """
    retrieved_at_k = retrieved[:k]
    relevant_count = sum(1 for context in retrieved_at_k if context in relevant)
    return relevant_count / k


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Calculate Recall@K.
    :param retrieved: List of retrieved contexts.
    :param relevant: List of ground truth relevant contexts.
    :param k: Number of top contexts to consider.
    :return: Recall@K value.
    """
    retrieved_at_k = retrieved[:k]
    relevant_count = sum(1 for context in retrieved_at_k if context in relevant)
    return relevant_count / len(relevant) if relevant else 0.0


def mean_reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    :param retrieved: List of retrieved contexts.
    :param relevant: List of ground truth relevant contexts.
    :return: MRR value.
    """
    for rank, context in enumerate(retrieved, start=1):
        if context in relevant:
            return 1 / rank
    return 0.0

# ----------------------- Metric Aggregation ----------------------


def evaluate_retrieval(queries: List[str], 
                       ground_truths: List[List[str]], 
                       top_k: int, 
                       retrieve_fn: Callable[[str, int], List[str]]) -> Dict[str, Any]:
    """
    Evaluate retrieval metrics over multiple queries.
    :param queries: List of queries.
    :param ground_truths: List of lists, where each inner list contains relevant contexts for a query.
    :param top_k: Number of top contexts to retrieve.
    :param retrieve_fn: Function to retrieve contexts for a given query.
    :return: Dictionary of aggregated metrics (Precision@K, Recall@K, MRR).
    """
    precisions = []
    recalls = []
    mrrs = []
    
    for query, relevant in zip(queries, ground_truths):
        retrieved = retrieve_fn(query, top_k=top_k)
        precisions.append(precision_at_k(retrieved, relevant, k=top_k))
        recalls.append(recall_at_k(retrieved, relevant, k=top_k))
        mrrs.append(mean_reciprocal_rank(retrieved, relevant))
    
    # Aggregate metrics
    return {
        f"Precision@{top_k}": sum(precisions) / len(queries),
        f"Recall@{top_k}": sum(recalls) / len(queries),
        "MRR": sum(mrrs) / len(queries),
    }

import math
from typing import List, Tuple, Any, Dict, Optional


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Computes cosine similarity between two float vectors.
    Returns a score between -1.0 and 1.0 (clamped to [0.0, 1.0] for similarity ranking).
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a, b in zip(vec_a, vec_a)))
    norm_b = math.sqrt(sum(b * b for b, b in zip(vec_b, vec_b)))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    sim = dot_product / (norm_a * norm_b)
    # Clamp to [0.0, 1.0] for semantic similarity representation
    return max(0.0, min(1.0, float(sim)))


class SimilaritySearchService:
    """
    Identifies semantically related requirement pairs based on vector embeddings
    and cosine similarity thresholds.
    """

    def find_candidate_pairs(
        self,
        requirements: List[Any],
        threshold: float = 0.50,
        cross_document_only: bool = False,
        max_candidates: int = 100
    ) -> List[Tuple[Any, Any, float]]:
        """
        Takes a list of Requirement model instances (each with get_embedding() method)
        and finds all pairs whose cosine similarity meets or exceeds `threshold`.

        Returns:
            List of tuples: (req_a, req_b, similarity_score)
            sorted by similarity_score descending.
        """
        n = len(requirements)
        if n < 2:
            return []

        # Pre-fetch embeddings
        embeddings = [req.get_embedding() for req in requirements]
        candidate_pairs = []

        for i in range(n):
            vec_a = embeddings[i]
            if not vec_a:
                continue

            doc_a_id = getattr(requirements[i], 'document_id', None)

            for j in range(i + 1, n):
                doc_b_id = getattr(requirements[j], 'document_id', None)

                # If cross-document only is requested, skip intra-document pairs
                if cross_document_only and doc_a_id is not None and doc_b_id is not None and doc_a_id == doc_b_id:
                    continue

                vec_b = embeddings[j]
                if not vec_b:
                    continue

                score = cosine_similarity(vec_a, vec_b)
                if score >= threshold:
                    candidate_pairs.append((requirements[i], requirements[j], round(score, 4)))

        # Sort candidate pairs by highest similarity first
        candidate_pairs.sort(key=lambda item: item[2], reverse=True)

        # Cap at max_candidates to prevent runaway LLM calls if documents are massive
        return candidate_pairs[:max_candidates]

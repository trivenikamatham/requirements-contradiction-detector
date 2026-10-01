import abc
import hashlib
import logging
import math
import re
from collections import Counter
from typing import List, Optional
from django.conf import settings

logger = logging.getLogger(__name__)


class EmbeddingProviderError(Exception):
    """Raised when an embedding provider fails to generate vectors."""
    pass


class BaseEmbeddingProvider(abc.ABC):
    """
    Abstract interface for generating semantic vector embeddings.
    Allows swappable providers (OpenAI, Local TF-IDF, Safe Mock).
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Returns the human-readable provider name."""
        pass

    @abc.abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates vector embeddings for a list of requirement texts.
        Returns a list of float vectors, each normalized to unit length.
        """
        pass

    def get_embedding(self, text: str) -> List[float]:
        """Convenience method for a single text."""
        return self.get_embeddings([text])[0]


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    Generates embeddings using OpenAI API (e.g. text-embedding-3-small).
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or getattr(settings, 'OPENAI_API_KEY', '')
        self.model = model or getattr(settings, 'OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
        
        if not self.api_key:
            raise EmbeddingProviderError("OPENAI_API_KEY is not configured.")

    @property
    def name(self) -> str:
        return f"OpenAI Embeddings ({self.model})"

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            batch_size = 100
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                cleaned_batch = [t.replace("\n", " ") for t in batch]
                response = client.embeddings.create(input=cleaned_batch, model=self.model)
                embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings)

            return all_embeddings
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise EmbeddingProviderError(f"OpenAI embedding call failed: {str(e)}")


class LocalTfidfEmbeddingProvider(BaseEmbeddingProvider):
    """
    Generates semantic vector embeddings using pure-Python sublinear TF-IDF
    with unigram + bigram n-grams, stopword filtering, and L2 unit-norm vectors.
    - 100% offline, requires NO external API keys or heavy OpenMP binary extensions.
    - Robust against Windows Python 3.13 multithreading deadlock issues.
    - Normalized dense vectors fully compatible with cosine similarity.
    """

    def __init__(self, max_features: int = 512):
        self.max_features = max_features
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'to', 'in', 'on', 'of', 'for', 'with',
            'by', 'at', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been',
            'shall', 'must', 'will', 'that', 'this', 'these', 'those', 'it', 'its',
            'into', 'upon', 'than', 'such', 'per', 'via', 'any', 'all', 'both'
        }

    @property
    def name(self) -> str:
        return "Local TF-IDF Semantic Vectors (Offline)"

    def _tokenize(self, text: str) -> List[str]:
        words = [
            w for w in re.findall(r'[a-zA-Z0-9_\-]+', text.lower())
            if w not in self.stop_words and len(w) > 1
        ]
        # Include bigrams to capture compound technical terminology
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
        return words + bigrams

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        doc_tokens = [self._tokenize(t) for t in texts]
        n_docs = len(texts)

        # Document frequency count
        df = Counter()
        for tokens in doc_tokens:
            df.update(set(tokens))

        # Select top features
        most_common = df.most_common(self.max_features)
        vocab = {term: idx for idx, (term, _) in enumerate(most_common)}
        vocab_size = max(len(vocab), 1)

        vectors = []
        for tokens in doc_tokens:
            tf = Counter(tokens)
            vec = [0.0] * vocab_size
            for term, count in tf.items():
                if term in vocab:
                    idx = vocab[term]
                    idf = math.log((1.0 + n_docs) / (1.0 + df[term])) + 1.0
                    sublinear_tf = 1.0 + math.log(count)
                    vec[idx] = sublinear_tf * idf

            # L2 normalization to unit vector
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [round(x / norm, 6) for x in vec]
            vectors.append(vec)

        return vectors


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """
    Deterministic hash-based mock vectorizer.
    Guarantees that similar texts share vector components,
    useful for zero-dependency unit tests and testing environments.
    """

    def __init__(self, vector_dim: int = 128):
        self.vector_dim = vector_dim

    @property
    def name(self) -> str:
        return "Deterministic Mock Vectorizer"

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            tokens = text.lower().split()
            vec = [0.0] * self.vector_dim

            for token in tokens:
                h = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)
                idx = h % self.vector_dim
                sign = 1.0 if ((h >> 8) & 1) else -1.0
                vec[idx] += sign

            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [round(x / norm, 6) for x in vec]
            results.append(vec)

        return results


def get_embedding_provider(provider_type: Optional[str] = None) -> BaseEmbeddingProvider:
    """
    Factory function to retrieve the configured embedding provider.
    Gracefully falls back to LocalTfidfEmbeddingProvider if OpenAI API key is missing.
    """
    provider_name = (provider_type or getattr(settings, 'EMBEDDING_PROVIDER', 'local')).lower()
    openai_key = getattr(settings, 'OPENAI_API_KEY', '')

    if provider_name == 'openai':
        if openai_key and openai_key.strip():
            try:
                return OpenAIEmbeddingProvider()
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAIEmbeddingProvider: {e}. Falling back to Local TF-IDF.")
                return LocalTfidfEmbeddingProvider()
        else:
            logger.info("OPENAI_API_KEY is not set. Using LocalTfidfEmbeddingProvider.")
            return LocalTfidfEmbeddingProvider()

    elif provider_name == 'mock':
        return MockEmbeddingProvider()

    else:
        return LocalTfidfEmbeddingProvider()

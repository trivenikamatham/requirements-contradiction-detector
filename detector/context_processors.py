from django.conf import settings

def provider_status(request):
    """
    Context processor to safely display active AI and embedding providers
    to users without exposing sensitive credentials or keys.
    """
    llm_provider = getattr(settings, 'LLM_PROVIDER', 'mock')
    openai_key_present = bool(getattr(settings, 'OPENAI_API_KEY', ''))
    groq_key_present = bool(getattr(settings, 'GROQ_API_KEY', ''))
    
    if llm_provider == 'openai':
        if openai_key_present:
            llm_display = f"OpenAI ({getattr(settings, 'OPENAI_MODEL', 'gpt-4o-mini')})"
            llm_badge = "success"
        else:
            llm_display = "Safe Mock Evaluator (OpenAI key not set)"
            llm_badge = "warning"
    elif llm_provider == 'groq':
        if groq_key_present:
            llm_display = f"Groq ({getattr(settings, 'GROQ_MODEL', 'llama-3.3-70b-versatile')})"
            llm_badge = "success"
        else:
            llm_display = "Safe Mock Evaluator (Groq key not set)"
            llm_badge = "warning"
    else:
        llm_display = "Safe Heuristic LLM Mock (Offline Ready)"
        llm_badge = "info"

    emb_provider = getattr(settings, 'EMBEDDING_PROVIDER', 'local')
    if emb_provider == 'openai' and openai_key_present:
        emb_display = f"OpenAI Embeddings ({getattr(settings, 'OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')})"
    elif emb_provider == 'openai' and not openai_key_present:
        emb_display = "Local TF-IDF (OpenAI key not set, auto-fallback)"
    elif emb_provider == 'local':
        emb_display = "Local Scikit-Learn TF-IDF Vectors"
    else:
        emb_display = "Mock Deterministic Vectors"

    return {
        'active_llm_display': llm_display,
        'active_llm_badge': llm_badge,
        'active_embedding_display': emb_display,
        'default_similarity_threshold': getattr(settings, 'DEFAULT_SIMILARITY_THRESHOLD', 0.50),
        'max_upload_size_mb': getattr(settings, 'MAX_UPLOAD_SIZE_MB', 15),
    }

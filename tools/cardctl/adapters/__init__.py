# tools/cardctl/adapters/__init__.py
"""Provider adapter registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseAdapter

# Lazy registry: adapters imported on demand to avoid heavy imports.
_ADAPTER_REGISTRY: dict[str, str] = {
    "openai": "openai_adapter.OpenAIAdapter",
    "anthropic": "anthropic_adapter.AnthropicAdapter",
    "google": "google_adapter.GoogleAdapter",
    "mistral": "mistral_adapter.MistralAdapter",
    "deepseek": "deepseek_adapter.DeepSeekAdapter",
    "xai": "xai_adapter.XAIAdapter",
    "qwen": "qwen_adapter.QwenAdapter",
    "moonshot": "moonshot_adapter.MoonshotAdapter",
    "ollama": "ollama_adapter.OllamaAdapter",
    "openrouter": "openrouter_adapter.OpenRouterAdapter",
    "poe": "poe_adapter.PoeAdapter",
    "huggingface": "huggingface_adapter.HuggingFaceAdapter",
}


def get_adapter(provider: str, **kwargs) -> "BaseAdapter":
    """Instantiate an adapter by provider name."""
    if provider not in _ADAPTER_REGISTRY:
        raise ValueError(
            f"No adapter for provider '{provider}'. "
            f"Available: {', '.join(sorted(_ADAPTER_REGISTRY))}"
        )
    module_class = _ADAPTER_REGISTRY[provider]
    module_name, class_name = module_class.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(f".{module_name}", package=__package__)
    cls = getattr(mod, class_name)
    return cls(**kwargs)


def list_providers() -> list[str]:
    """Return sorted list of all supported provider names."""
    return sorted(_ADAPTER_REGISTRY.keys())

"""OWASP LLM Top-10 risk category identifiers for HITL metadata."""

from __future__ import annotations

from enum import Enum


class OwaspLlmRisk(str, Enum):
    """OWASP LLM Top-10 (2025 edition) risk categories."""

    LLM01_PROMPT_INJECTION = "LLM01_prompt_injection"
    LLM02_INSECURE_OUTPUT = "LLM02_insecure_output_handling"
    LLM03_TRAINING_DATA_POISONING = "LLM03_training_data_poisoning"
    LLM04_MODEL_DENIAL_OF_SERVICE = "LLM04_model_dos"
    LLM05_SUPPLY_CHAIN_VULNERABILITIES = "LLM05_supply_chain"
    LLM06_EXCESSIVE_AGENCY = "LLM06_excessive_agency"
    LLM07_SYSTEM_PROMPT_LEAKAGE = "LLM07_system_prompt_leakage"
    LLM08_VECTOR_EMBEDDING_WEAKNESSES = "LLM08_vector_embedding_weaknesses"
    LLM09_MISINFORMATION = "LLM09_misinformation"
    LLM10_UNBOUNDED_CONSUMPTION = "LLM10_unbounded_consumption"


OWASP_LLM_RISK_VALUES: tuple[str, ...] = tuple(risk.value for risk in OwaspLlmRisk)


__all__ = ["OWASP_LLM_RISK_VALUES", "OwaspLlmRisk"]

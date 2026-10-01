# src/llmcore/routing/verifiers/__init__.py
"""Response verifiers, for cascades.

Classification guesses difficulty *before* seeing the answer. A cascade checks
*after*: answer cheaply, judge the answer, escalate only if it fell short.
FrugalGPT (arXiv:2305.05176) reports large savings from exactly this shape, and
llmcore is unusually well placed to do it because it already has the cheap
target, the expensive target and the accounting in one place.

The judgement a verifier returns is three-valued, and that matters more than
it looks:

* ``sufficient=True`` — keep the cheap answer.
* ``sufficient=False`` — escalate.
* ``sufficient=None`` — **could not judge.** Not a pass, not a fail.

Folding ``None`` into either one is the bug that makes cascades useless.
Reading it as a fail escalates every unjudgeable answer, which inverts the
cost saving the cascade exists for; reading it as a pass silently disables the
quality floor the moment the verifier breaks. So it stays a third value, and
what to do about it is the user's configured choice
(``routing.cascade.on_unknown``, default ``accept``).

Available verifiers:

===============  =============================================  ======
Name             Mechanism                                      Cost
===============  =============================================  ======
``script``       a user callable: compile it, parse it, test it  free
``typesafe_jev`` a TypeSafe ``noul`` question                    api
``llm``          any cheap target as a judge                     api
===============  =============================================  ======

``script`` is the strongest of the three in practice, and it is worth saying
why: "does this code compile", "does this JSON match the schema", "do the
tests pass" are cheap, exact, and better than any model's opinion. A cascade
whose verifier is a compiler is not a heuristic at all.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Mapping

if TYPE_CHECKING:
    from ..protocols import ResponseVerifier

logger = logging.getLogger(__name__)

__all__ = [
    "available_verifiers",
    "build_verifier",
    "register_verifier",
]

_FACTORIES: dict[str, Callable[..., ResponseVerifier]] = {}


def register_verifier(
    name: str, factory: Callable[..., ResponseVerifier], *, replace: bool = False
) -> None:
    """Register a verifier factory under ``name``."""
    key = name.strip().lower()
    if key in _FACTORIES and not replace:
        raise ValueError(f"Verifier '{key}' is already registered; pass replace=True to swap it.")
    _FACTORIES[key] = factory


def available_verifiers() -> list[str]:
    """Return every registered verifier name."""
    return sorted(_FACTORIES)


def build_verifier(
    name: str, *, config: Mapping[str, Any] | None = None
) -> ResponseVerifier | None:
    """Build one verifier by name, or ``None`` if it cannot be built.

    A missing verifier disables the cascade rather than failing the request:
    without a judge there is nothing to escalate *on*, and answering from the
    cheap rung is a strictly better outcome than refusing to answer. The log
    says so plainly, because a cascade that is quietly not running is a
    cost-saving the user thinks they have and does not.
    """
    key = name.strip().lower()
    factory = _FACTORIES.get(key)
    if factory is None:
        logger.warning(
            "Unknown verifier '%s'; the cascade will not escalate. Available: %s",
            name,
            ", ".join(available_verifiers()),
        )
        return None
    try:
        return factory(config=dict(config or {}))
    except Exception as exc:
        logger.warning(
            "Verifier '%s' could not be built (%s); the cascade will not escalate.", key, exc
        )
        return None


def _register_builtins() -> None:
    from . import simple  # noqa: F401

    for module in ("typesafe_noul", "llm_judge"):
        try:
            __import__(f"{__name__}.{module}")
        except Exception:  # pragma: no cover - optional surfaces
            logger.debug("Verifier module %s is unavailable", module, exc_info=True)


_register_builtins()

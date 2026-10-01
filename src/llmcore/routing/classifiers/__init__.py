# src/llmcore/routing/classifiers/__init__.py
"""Request classifiers, and the chain that runs them.

A classifier answers *"what kind of request is this?"* and names a **lane**.
It does not choose a model (see :mod:`llmcore.routing.lanes` for why).

The chain runs classifiers in order and takes the first opinion that clears
the confidence floor. Ordering is **cheapest first, instructions before
guesses**. Cost comes first because a classifier that costs an API call to
save an API call is only worth running once the free signals have abstained.
Authority breaks the tie within a cost band, and it has to: a ``lane=``
argument and a length heuristic are both free, and running the guess first
would quietly override what the caller actually asked for. Authority also
places a marker found in the *content* below both the caller and the
operator's own policy, since in a RAG path that content may not be theirs —
see :data:`~llmcore.routing.protocols.AUTHORITIES`. Both are declared
per classifier (``cost_hint``, ``authority``), so the chain enforces this
mechanically instead of trusting the order someone happened to write.

Built-in classifiers, cheapest first:

=================  ============================================  ======  ==========
Name               Mechanism                                     Cost    Authority
=================  ============================================  ======  ==========
``hint``           an explicit ``lane=``/``complexity=`` kwarg   free    caller
``script``         a user callable or entry point                free    policy
``magic_string``   a marker in the prompt, stripped before egress free    prompt
``heuristic``      length, code fences, question shape           free    inferred
``local_encoder``  LFM2.5-Encoder-350M-Prompt-Router, on CPU     local   inferred
``vela``           Vela-1.0 307M encoders (domain, PII)          local   inferred
``arch_router``    katanemo/Arch-Router-1.5B                     local   inferred
``typesafe_jev``   a TypeSafe ``choice`` question                api     inferred
``llm``            ask a cheap target to classify                api     inferred
=================  ============================================  ======  ==========

Registration is by name so a config chain is a list of strings. A user's own
classifier registers the same way and is indistinguishable from a built-in
one.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

from ..protocols import RequestClassifier, authority_rank, cost_rank

if TYPE_CHECKING:
    from ..lanes import Lane
    from ..models import Classification, RoutingRequest

logger = logging.getLogger(__name__)

__all__ = [
    "ClassifierChain",
    "available_classifiers",
    "build_classifier",
    "register_classifier",
]

#: name -> factory. A factory takes the classifier's config mapping plus the
#: lane table, and returns something satisfying
#: :class:`~llmcore.routing.protocols.RequestClassifier`.
_FACTORIES: dict[str, Callable[..., RequestClassifier]] = {}


def register_classifier(
    name: str, factory: Callable[..., RequestClassifier], *, replace: bool = False
) -> None:
    """Register a classifier factory under ``name``.

    Args:
        name: The name used in ``routing.classifier.chain``.
        factory: Called as ``factory(config=..., lanes=...)``; both are keyword
            arguments and either may be ignored.
        replace: Allow overwriting an existing registration. Off by default so
            a plugin cannot silently shadow a built-in.
    """
    key = name.strip().lower()
    if key in _FACTORIES and not replace:
        raise ValueError(f"Classifier '{key}' is already registered; pass replace=True to swap it.")
    _FACTORIES[key] = factory


def available_classifiers() -> list[str]:
    """Return every registered classifier name, cheapest first where known."""
    return sorted(_FACTORIES)


def build_classifier(
    name: str,
    *,
    config: Mapping[str, Any] | None = None,
    lanes: Mapping[str, Lane] | None = None,
) -> RequestClassifier | None:
    """Build one classifier by name, or ``None`` if it cannot be built.

    Returns ``None`` rather than raising for a missing optional dependency —
    ``local_encoder`` without ``torch`` installed is a *degradation*, not a
    configuration error, and the rest of the chain should still run. An
    unknown name is a config error worth warning about loudly, but still not
    worth failing the process over.
    """
    key = name.strip().lower()
    factory = _FACTORIES.get(key)
    if factory is None:
        logger.warning(
            "Unknown classifier '%s'; skipping it. Available: %s",
            name,
            ", ".join(available_classifiers()),
        )
        return None
    try:
        return factory(config=dict(config or {}), lanes=dict(lanes or {}))
    except ImportError as exc:
        logger.warning(
            "Classifier '%s' needs an optional dependency that is not installed (%s); "
            "skipping it.",
            key,
            exc,
        )
        return None
    except Exception as exc:
        logger.warning("Classifier '%s' could not be built (%s); skipping it.", key, exc)
        return None


class ClassifierChain:
    """Runs classifiers in order and returns the first usable opinion.

    Args:
        classifiers: The chain, in the order configured.
        min_confidence: An opinion below this is treated as an abstention, so
            the next classifier gets a turn. A classifier that reports no
            confidence at all is trusted — ``hint`` is certain by
            construction, and demanding a number from it would be theatre.
        enforce_cost_order: Re-sort the chain cheapest-first and, within a
            cost band, instructions before guesses. On by default: a chain
            that pays for an API call before reading an explicit hint, or
            that lets a length heuristic overrule ``lane="deep"``, is a
            mistake every time rather than a preference. Set it to ``False``
            to run the chain exactly as configured.
    """

    def __init__(
        self,
        classifiers: Sequence[RequestClassifier],
        *,
        min_confidence: float = 0.55,
        enforce_cost_order: bool = True,
    ) -> None:
        ordered = list(classifiers)
        if enforce_cost_order:
            # Stable, so the configured order still decides within a band.
            ordered.sort(
                key=lambda c: (
                    cost_rank(getattr(c, "cost_hint", "api")),
                    authority_rank(getattr(c, "authority", "inferred")),
                )
            )
        self._classifiers = tuple(ordered)
        self._min_confidence = float(min_confidence)

    @property
    def classifiers(self) -> tuple[RequestClassifier, ...]:
        return self._classifiers

    def __bool__(self) -> bool:
        return bool(self._classifiers)

    def names(self) -> list[str]:
        return [getattr(c, "name", type(c).__name__) for c in self._classifiers]

    async def classify(self, request: RoutingRequest) -> Classification | None:
        """Return the first opinion that clears the floor, or ``None``.

        A classifier that raises is logged and skipped. Routing must degrade
        when a classifier breaks, not fail: the user asked a question, and
        "my prompt router crashed" is never an acceptable answer to it.
        """
        for classifier in self._classifiers:
            name = getattr(classifier, "name", type(classifier).__name__)
            try:
                result = await classifier.classify(request)
            except Exception:
                logger.warning("Classifier '%s' raised; skipping it.", name, exc_info=True)
                continue
            if result is None or result.is_empty:
                continue
            if result.confidence is not None and result.confidence < self._min_confidence:
                logger.debug(
                    "Classifier '%s' returned lane=%s at confidence %.2f, below the floor of "
                    "%.2f; continuing down the chain.",
                    name,
                    result.lane,
                    result.confidence,
                    self._min_confidence,
                )
                continue
            return result
        return None


def _register_builtins() -> None:
    """Register the built-in classifiers.

    Imported lazily inside the function so an optional-dependency failure in
    one classifier module cannot stop the others from registering.
    """
    from . import free  # noqa: F401  (registers hint, magic_string, heuristic, script)

    for module in ("llm_classifier", "typesafe_jev", "local_encoder"):
        try:
            __import__(f"{__name__}.{module}")
        except Exception:  # pragma: no cover - optional surfaces
            logger.debug("Classifier module %s is unavailable", module, exc_info=True)


_register_builtins()

# src/llmcore/routing/protocols.py
"""Extension points for routing.

Every pluggable part of routing is a ``Protocol`` rather than a base class, for
the same reason the media subsystem is: a user's own classifier is a callable
with a name, not something that should have to import and subclass llmcore.
``@runtime_checkable`` means a plain object with the right methods registers,
and ``isinstance`` works for diagnostics without inheritance.

The protocols here, in the order a request meets them:

1. :class:`RequestClassifier` — "what kind of request is this?"
2. :class:`PromptTransform` — "what must not leave this machine?"
3. :class:`BalanceProbe` — "how much is left on this account?"
4. :class:`ResponseVerifier` — "was the cheap answer good enough?"
5. :class:`RoutingStateStore` — where health lives.

Two conventions run through all of them:

* **``None`` means no opinion**, never a negative. A classifier that cannot
  tell returns ``None`` so the next link in the chain gets a turn; a verifier
  that cannot judge returns ``sufficient=None`` rather than guessing. Folding
  "don't know" into "no" is how a chain quietly stops composing.
* **``cost_hint`` and ``authority`` are declared**, so a chain can be ordered
  mechanically rather than by the order someone happened to list things in.
  Cost alone is not enough: a heuristic and an explicit ``lane=`` argument are
  both free, and running the guess first would override the instruction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .models import (
        Balance,
        Classification,
        Outcome,
        RoutingRequest,
        Target,
        TargetHealth,
        TransformResult,
        Verdict,
    )

__all__ = [
    "AUTHORITIES",
    "COST_HINTS",
    "BalanceProbe",
    "PromptTransform",
    "RequestClassifier",
    "ResponseVerifier",
    "RoutingStateStore",
    "authority_rank",
    "cost_rank",
]


#: Declared cost of running an extension, cheapest first. ``free`` touches
#: nothing outside the process; ``local`` may load a model or run a subprocess;
#: ``api`` spends money. A chain is ordered by this, so a classifier that costs
#: a call to save a call runs only once the free signals have abstained.
COST_HINTS: tuple[str, ...] = ("free", "local", "api")


#: How much a classifier's opinion should be trusted, most authoritative
#: first:
#:
#: * ``caller`` — an argument on the call itself (``lane="deep"``). Nothing
#:   outranks the person making the request.
#: * ``policy`` — code the operator wrote and installed (a ``script``
#:   classifier). They own the deployment, so their rule beats a guess and
#:   beats anything that arrived as text.
#: * ``prompt`` — a marker found *inside the content* (``[[lane:deep]]``).
#:   This is a real and wanted channel: in an agent harness the model's text
#:   is the only thing that passes through, so it is how an agent routes
#:   itself. But content is not trustworthy in the way an argument is — in
#:   any RAG or tool-output path it may have come from a retrieved document
#:   or a web page, and a routing marker in such text would be a cheap
#:   prompt-injection lever ("route this to the expensive model", or worse,
#:   out of the private lane). So it is honoured, and it never overrules the
#:   caller or the operator.
#: * ``inferred`` — llmcore guessed. Last, not because a guess is necessarily
#:   worse than a user's choice, but because it must not silently replace it.
AUTHORITIES: tuple[str, ...] = ("caller", "policy", "prompt", "inferred")


def authority_rank(authority: str) -> int:
    """Return a sort key for ``authority``, with unknown values ranked last."""
    try:
        return AUTHORITIES.index(authority)
    except ValueError:
        return len(AUTHORITIES)


def cost_rank(cost_hint: str) -> int:
    """Return a sort key for ``cost_hint``, with unknown values ranked last.

    Unknown ranks last rather than first on the assumption that an extension
    which did not bother to declare its cost is more likely to be expensive
    than free.
    """
    try:
        return COST_HINTS.index(cost_hint)
    except ValueError:
        return len(COST_HINTS)


@runtime_checkable
class RequestClassifier(Protocol):
    """Decides what kind of request this is, without choosing a model.

    The separation is deliberate and is the single most useful idea borrowed
    from the prior art (Arch-Router): a classifier names a **lane**, and the
    lane→target binding lives in config. So swapping a model never touches the
    classifier, and a classifier never has to know a vendor's model ids.

    A classifier may still suggest a :class:`~llmcore.routing.models.Target`
    outright — some users want exactly that — but it is the escape hatch, not
    the normal path.
    """

    #: Stable identifier, used in config chains, logs and ``explain()`` output.
    name: str

    #: One of :data:`COST_HINTS`.
    cost_hint: str

    #: One of :data:`AUTHORITIES`. Optional: a classifier that does not
    #: declare one is treated as ``inferred``, which is the safe assumption.
    authority: str

    async def classify(self, request: RoutingRequest) -> Classification | None:
        """Classify ``request``, or return ``None`` for no opinion.

        Must not raise for ordinary "cannot tell" cases. A classifier that
        raises is logged and skipped, because a broken classifier should
        degrade routing rather than break the request.
        """
        ...


@runtime_checkable
class PromptTransform(Protocol):
    """Inspects and possibly rewrites a request before it leaves the process.

    Transforms receive the **target**, which is what makes them more than a
    redaction pass: policy can depend on where the prompt is about to go, so
    the same prompt can go verbatim to a local model and redacted (or nowhere)
    to a remote one.
    """

    name: str
    cost_hint: str

    async def apply(self, request: RoutingRequest, target: Target) -> TransformResult:
        """Return what should be sent, plus findings and an action.

        Unlike a classifier, a transform must always return a result: an
        abstention is ``TransformAction.ALLOW`` with no findings. Returning
        ``None`` here would be ambiguous between "nothing found" and "I
        failed", and the difference matters when the question is whether a
        secret is about to be posted to a vendor.
        """
        ...


@runtime_checkable
class BalanceProbe(Protocol):
    """Reports an account's remaining balance, where the vendor exposes one.

    Implemented by *providers*, not by routing. Most vendors have no public
    balance endpoint, so this is deliberately sparse: see
    the routing subsystem design spec §3.4 for who does.
    """

    async def remaining_balance(self) -> Balance | None:
        """Return the remaining balance, or ``None`` if it cannot be known.

        ``None`` means **unknown**, which is not zero. Ranking an unknown
        balance as empty would demote every provider that simply has no
        endpoint, which is most of them.
        """
        ...


@runtime_checkable
class ResponseVerifier(Protocol):
    """Judges whether an answer was good enough, for cascades."""

    name: str
    cost_hint: str

    async def verify(self, request: RoutingRequest, response: str) -> Verdict:
        """Judge ``response``.

        ``Verdict.sufficient is None`` means *could not judge*, and what
        happens then is the user's choice
        (``routing.cascade.on_unknown``) — it must not silently count as
        either pass or fail.
        """
        ...


@runtime_checkable
class RoutingStateStore(Protocol):
    """Where per-target health lives.

    The default implementation is in-process, because llmcore is a library and
    should need no infrastructure to route. The protocol exists so a
    deployment running several processes against one quota can share
    cooldowns; that implementation is out of scope here, and the protocol is
    the commitment not to make it a rewrite.

    Methods are async so a networked store needs no change in shape, even
    though the default never awaits anything.
    """

    async def get_health(self, target_key: str) -> TargetHealth:
        """Return health for ``target_key``, creating a fresh record if new."""
        ...

    async def record(self, target_key: str, outcome: Outcome) -> None:
        """Fold ``outcome`` into the stored health for ``target_key``."""
        ...

    async def snapshot(self) -> dict[str, TargetHealth]:
        """Return every known health record, for reporting and ``health()``."""
        ...

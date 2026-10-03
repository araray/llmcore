# src/llmcore/runtimes/sizing.py
"""Sizing: will this model fit, on what, and at what context (spec phase R2).

Everything here is **read-only and free**. Nothing in this module can start
billable compute, which is the whole reason sizing is a separate phase from
provisioning: a caller has to be able to find out what a launch would cost
before being asked to approve it.

The arithmetic is deliberately printed rather than hidden. A sizer that answers
"use an A100" and shows nothing is impossible to argue with, and the first
thing anyone wants to know is *why* a 30B model needs more than 24 GB. So every
:class:`~llmcore.runtimes.models.Plan` carries its own working.

Three facts drive the estimate:

* **Weights** — exact parameter counts come from the Hub's ``safetensors``
  metadata, which reports counts per dtype. Guessing from file sizes is the
  fallback, not the plan.
* **KV cache** — ``2 x layers x kv_heads x head_dim x bytes x ctx``. The
  ``kv_heads`` term is what makes grouped-query attention cheap, and getting it
  wrong is the single biggest sizing error available: reading
  ``num_attention_heads`` instead of ``num_key_value_heads`` overestimates
  Llama-3.3-70B's cache by **8x**.
* **Overhead** — activations, CUDA graphs and allocator fragmentation, plus
  vLLM's own ``gpu_memory_utilization`` ceiling.

When nothing fits, the sizer **refuses with a concrete alternative** rather
than returning a plan that will fail on the VM after the money starts.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .models import CostUnit, ModelSpec, Plan, Quantization

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_SKU_LADDER",
    "GPU_SKUS",
    "SKU_ALIASES",
    "GpuSku",
    "Sizer",
    "bytes_per_param",
    "detect_quantization",
    "resolve_sku",
]


@dataclass(frozen=True, slots=True)
class GpuSku:
    """One rung on the ladder.

    Attributes:
        name: SKU name used in config, plans and handles.
        vram_gb: Device VRAM assumed when sizing.
        cli_gpu: What to pass to the backend CLI's ``--gpu``. Separate from
            ``name`` because the Colab CLI accepts only ``T4, L4, G4, H100,
            A100`` -- it cannot express "the 80 GB A100", so a plan must not
            promise one.
        high_mem: Whether this rung needs ``--high-mem`` (system RAM, not
            VRAM).
        needs_fp16: The card predates Ampere and has no bfloat16. vLLM does
            **not** silently downcast -- it refuses with "Bfloat16 is only
            supported on GPUs with compute capability of at least 8.0" -- so a
            recipe serving a bf16 checkpoint here must be told ``--dtype
            half`` explicitly.
        cost_per_hour: Indicative burn rate in :attr:`cost_unit`, for the
            spend summary. Colab bills in compute units rather than currency,
            and quoting a dollar figure llmcore cannot verify would be worse
            than quoting the unit the user is actually charged in -- which is
            why the unit travels with the number.
        cost_unit: What ``cost_per_hour`` is denominated in.
        gpu_count: Devices in this rung. ``1`` for Colab, which sells no other
            shape. Rental catalogues list multi-GPU nodes as separate rows, and
            on those ``vram_gb`` is the **aggregate** across all devices: vLLM
            shards weights and KV cache across a tensor-parallel group, so the
            group's combined VRAM is what a model is sized against.
        region: Where this rung is offered, for catalogues whose price varies
            by region. ``None`` when the backend has no region concept.
        offering_id: Backend-side id for this exact catalogue row, so a launch
            can be pinned to the row that was priced.
        notes: Anything a caller should know before picking it.
    """

    name: str
    vram_gb: float
    cli_gpu: str = ""
    high_mem: bool = False
    cost_per_hour: float | None = None
    cost_unit: CostUnit | None = None
    needs_fp16: bool = False
    gpu_count: int = 1
    region: str | None = None
    offering_id: str | None = None
    notes: str = ""

    @property
    def compute_units_per_hour(self) -> float | None:
        """Deprecated alias for :attr:`cost_per_hour` on Colab-billed rungs.

        Returns ``None`` for a rung priced in anything other than compute
        units, so a caller that still reads this name cannot be handed dollars
        under a field that promises compute units.
        """
        if self.cost_unit is not None and self.cost_unit is not CostUnit.COMPUTE_UNIT:
            return None
        return self.cost_per_hour

    @property
    def gpu_flag(self) -> str:
        """The value for the CLI's ``--gpu``."""
        return self.cli_gpu or self.name


#: Known SKUs, cheapest first.
#:
#: The names match what ``colab new --gpu`` actually accepts, which the draft
#: spec's ladder did not: it listed ``A100-40`` and ``A100-80`` as separate
#: rungs, and the CLI has one ``A100`` with no way to ask for a particular
#: amount of VRAM. A100 is therefore sized at the **conservative** 40 GB -- if
#: the account is given an 80 GB card the plan simply has more headroom than it
#: promised, which is the safe direction to be wrong in.
GPU_SKUS: dict[str, GpuSku] = {
    "T4": GpuSku("T4", 16.0, cost_per_hour=1.96, cost_unit=CostUnit.COMPUTE_UNIT,
                 needs_fp16=True,
                 notes="pre-Ampere: no bf16, so the recipe is told --dtype half"),
    "L4": GpuSku("L4", 24.0, cost_per_hour=4.82, cost_unit=CostUnit.COMPUTE_UNIT),
    "G4": GpuSku("G4", 24.0, cost_per_hour=4.82, cost_unit=CostUnit.COMPUTE_UNIT,
                 notes="L4-class"),
    "A100": GpuSku("A100", 40.0, cost_per_hour=11.77, cost_unit=CostUnit.COMPUTE_UNIT,
                   notes="Colab may assign a 40 GB or 80 GB card; sized for 40"),
    "H100": GpuSku("H100", 80.0, cost_per_hour=23.0, cost_unit=CostUnit.COMPUTE_UNIT),
}

#: Accepted spellings that are not SKU names, so a config written against the
#: draft spec keeps working instead of silently falling back to the default
#: ladder.
SKU_ALIASES: dict[str, str] = {
    "a100-40": "A100",
    "a100_40": "A100",
    "a100-80": "A100",
    "a100_80": "A100",
    "a100": "A100",
    "t4": "T4",
    "l4": "L4",
    "g4": "G4",
    "h100": "H100",
}


def resolve_sku(name: str) -> str | None:
    """Map a configured SKU spelling to a known SKU name, or ``None``."""
    key = str(name).strip()
    if key in GPU_SKUS:
        return key
    return SKU_ALIASES.get(key.lower())


#: The order the sizer walks. Overridable from config, so an account that
#: cannot get A100s does not have to be told twice.
DEFAULT_SKU_LADDER: tuple[str, ...] = ("T4", "L4", "G4", "A100", "H100")

#: Bytes per parameter by quantization. ``NONE`` means "as the repo stores it",
#: which is resolved from the dtype rather than assumed.
_QUANT_BYTES: dict[Quantization, float] = {
    Quantization.NONE: 2.0,
    Quantization.FP8: 1.0,
    Quantization.INT8: 1.0,
    Quantization.AWQ: 0.5,
    Quantization.GPTQ: 0.5,
    Quantization.INT4: 0.5,
    # GGUF levels are not exactly n/8 bytes: the format keeps some tensors at
    # higher precision, so a "4-bit" model lands nearer 4.4 bits per weight.
    Quantization.GGUF: 0.68,
    Quantization.GGUF_Q4: 0.55,
    Quantization.GGUF_Q5: 0.68,
    Quantization.GGUF_Q8: 1.06,
}

#: Bytes per element for a dtype name as the Hub reports it.
_DTYPE_BYTES: dict[str, float] = {
    "F64": 8.0, "F32": 4.0, "BF16": 2.0, "F16": 2.0, "FP16": 2.0,
    "I64": 8.0, "I32": 4.0, "I16": 2.0, "I8": 1.0, "U8": 1.0,
    "F8_E4M3": 1.0, "F8_E5M2": 1.0, "F8": 1.0, "BOOL": 1.0,
    "I4": 0.5, "U4": 0.5,
}

#: Repo-name markers that imply a quantization. Checked in this order, because
#: a name can carry several and the most specific should win.
_QUANT_MARKERS: tuple[tuple[re.Pattern[str], Quantization], ...] = (
    (re.compile(r"\bawq\b", re.IGNORECASE), Quantization.AWQ),
    (re.compile(r"\bgptq\b", re.IGNORECASE), Quantization.GPTQ),
    (re.compile(r"\b(?:fp8|f8)\b", re.IGNORECASE), Quantization.FP8),
    (re.compile(r"\bq8(?:_\w+)?\b", re.IGNORECASE), Quantization.GGUF_Q8),
    (re.compile(r"\b(?:iq5|q5)(?:_\w+)?\b", re.IGNORECASE), Quantization.GGUF_Q5),
    (re.compile(r"\b(?:iq4|q4)(?:_\w+)?\b", re.IGNORECASE), Quantization.GGUF_Q4),
    (re.compile(r"\bint8\b", re.IGNORECASE), Quantization.INT8),
    (re.compile(r"\bint4\b", re.IGNORECASE), Quantization.INT4),
)


def bytes_per_param(quantization: Quantization) -> float:
    """Bytes per weight for *quantization*."""
    return _QUANT_BYTES.get(quantization, 2.0)


def detect_quantization(repo_id: str, *, config: dict[str, Any] | None = None) -> Quantization | None:
    """Infer quantization from a repo's config and name.

    Config wins over the name, because a repo that declares
    ``quantization_config`` is authoritative about itself while a name is a
    convention. Returns ``None`` when there is no evidence either way — which
    the caller must read as "unquantized", not as "unknown and therefore
    cheap".
    """
    declared = (config or {}).get("quantization_config")
    if isinstance(declared, dict):
        method = str(declared.get("quant_method") or declared.get("quantization_method") or "").lower()
        bits = declared.get("bits") or declared.get("w_bit")
        if "awq" in method:
            return Quantization.AWQ
        if "gptq" in method:
            return Quantization.GPTQ
        if "fp8" in method:
            return Quantization.FP8
        if "bitsandbytes" in method or "bnb" in method:
            return Quantization.INT4 if bits == 4 else Quantization.INT8
        if bits == 4:
            return Quantization.INT4
        if bits == 8:
            return Quantization.INT8

    # Match against the repo *name* only, not the owner: an organisation called
    # "awq-labs" does not make every model it publishes quantized.
    name = repo_id.rsplit("/", 1)[-1]
    # Treat '-' and '.' as word boundaries so "Q4_K_M" and "-AWQ" both match.
    haystack = re.sub(r"[-_.]", " ", name)
    for pattern, quantization in _QUANT_MARKERS:
        if pattern.search(haystack):
            return quantization
    return None


class Sizer:
    """Estimates what it takes to serve a model, without provisioning anything.

    Args:
        ladder: SKU names to consider, cheapest first.
        headroom_fraction: Fraction of a SKU's VRAM to keep free beyond the
            estimate. Defaults to 0.15, because an estimate that exactly fills
            a card OOMs on the first long request — and it OOMs *after* the
            money has started.
        gpu_memory_utilization: What vLLM will be told to use. The usable VRAM
            is the device total times this, not the device total.
        overhead_gb: Fixed allowance for activations, CUDA graphs and allocator
            fragmentation.
        hf_token: Hugging Face token for gated repos.
        offline: Skip the Hub entirely and size from local model cards. Useful
            on a machine with no network, and the reason sizing has a fallback
            chain at all.
        skus: The catalogue *ladder* names are resolved against. Defaults to
            :data:`GPU_SKUS`, which is Colab's fixed menu. A rental backend
            passes its own catalogue, read from its own API: its prices are
            live, per-region and in dollars, none of which a table in llmcore's
            source could keep honest.
        aliases: Extra accepted spellings for *skus*. Defaults to
            :data:`SKU_ALIASES`, and is empty when *skus* is supplied, since
            Colab's spellings mean nothing in another vendor's catalogue.
    """

    def __init__(
        self,
        *,
        ladder: tuple[str, ...] | list[str] = DEFAULT_SKU_LADDER,
        headroom_fraction: float = 0.15,
        gpu_memory_utilization: float = 0.90,
        overhead_gb: float = 2.5,
        hf_token: str | None = None,
        offline: bool = False,
        skus: Mapping[str, GpuSku] | None = None,
        aliases: Mapping[str, str] | None = None,
    ) -> None:
        self.skus: Mapping[str, GpuSku] = GPU_SKUS if skus is None else dict(skus)
        if aliases is not None:
            self.aliases: Mapping[str, str] = dict(aliases)
        elif skus is None:
            self.aliases = SKU_ALIASES
        else:
            self.aliases = {}
        self.ladder = tuple(ladder)
        self.headroom_fraction = float(headroom_fraction)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.overhead_gb = float(overhead_gb)
        self.hf_token = hf_token
        self.offline = offline

    # -- public -----------------------------------------------------------

    async def estimate(self, spec: ModelSpec) -> Plan:
        """Size *spec*. Free, read-only, and never provisions anything."""
        import asyncio

        return await asyncio.to_thread(self.estimate_sync, spec)

    def estimate_sync(self, spec: ModelSpec) -> Plan:
        """Synchronous :meth:`estimate`, for CLIs and tests."""
        notes: list[str] = []
        metadata = self._metadata(spec, notes)

        recipe = "llamacpp" if metadata.get("is_gguf") else "vllm"
        if recipe == "llamacpp":
            notes.append("GGUF repo detected; the llama.cpp recipe serves these, not vLLM")

        quantization = (
            spec.quantization
            or metadata.get("quantization")
            or Quantization.NONE
        )
        if spec.quantization is not None:
            notes.append(f"quantization {quantization} requested explicitly")
        elif metadata.get("quantization") is not None:
            notes.append(f"quantization {quantization} detected from the repo")

        weights_gb = self._weights_gb(metadata, quantization, notes)
        context = self._resolve_context(spec, metadata, notes)
        kv_gb = self._kv_gb(metadata, context, quantization, notes)
        required = weights_gb + kv_gb + self.overhead_gb
        notes.append(
            f"total = weights {weights_gb:.1f} + KV {kv_gb:.1f} + overhead "
            f"{self.overhead_gb:.1f} = {required:.1f} GB"
        )

        sku, usable, fits, context, kv_gb, required = self._choose_sku(
            metadata, quantization, weights_gb, context, required, notes
        )

        rung = self.skus.get(sku)
        plan = Plan(
            spec=spec if context == spec.context_length else _with_context(spec, context),
            sku=sku,
            recipe=recipe,
            quantization=quantization,
            vram_required_gb=round(required, 2),
            vram_available_gb=round(usable, 2),
            context_length=context,
            fits=fits,
            notes=tuple(notes),
            estimated_cost_per_hour=rung.cost_per_hour if rung else None,
            cost_unit=rung.cost_unit if rung else None,
            gpu_count=rung.gpu_count if rung else 1,
            region=rung.region if rung else None,
            offering_id=rung.offering_id if rung else None,
        )
        if not fits:
            return plan.with_notes(*self._refusal_advice(metadata, weights_gb, quantization))
        return plan

    # -- metadata ---------------------------------------------------------

    def _metadata(self, spec: ModelSpec, notes: list[str]) -> dict[str, Any]:
        """Collect what is known about the repo, cheapest source first."""
        metadata: dict[str, Any] = {"repo_id": spec.repo_id}

        if not self.offline:
            try:
                metadata.update(self._from_hub(spec, notes))
                return metadata
            except Exception as exc:
                notes.append(
                    f"Hub metadata unavailable ({_brief(exc)}); falling back to model cards"
                )
                logger.debug("Hub sizing metadata failed for %s", spec.repo_id, exc_info=True)

        metadata.update(self._from_cards(spec, notes))
        return metadata

    def _from_hub(self, spec: ModelSpec, notes: list[str]) -> dict[str, Any]:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi(token=self.hf_token)
        info = api.model_info(spec.repo_id, revision=spec.revision)
        out: dict[str, Any] = {}

        files = [sibling.rfilename for sibling in (info.siblings or [])]
        out["is_gguf"] = any(name.lower().endswith(".gguf") for name in files)

        safetensors = getattr(info, "safetensors", None)
        params_by_dtype = dict(getattr(safetensors, "parameters", None) or {})
        if params_by_dtype:
            out["params_by_dtype"] = params_by_dtype
            out["params"] = sum(params_by_dtype.values())
            notes.append(
                "parameters from the Hub safetensors index: "
                + ", ".join(f"{count:,} x {dtype}" for dtype, count in params_by_dtype.items())
            )
        elif getattr(safetensors, "total", None):
            out["params"] = int(safetensors.total)
            notes.append(f"parameters from the Hub index: {out['params']:,}")

        try:
            config_path = hf_hub_download(
                spec.repo_id,
                "config.json",
                revision=spec.revision,
                token=self.hf_token,
            )
            out["config"] = json.loads(open(config_path).read())
        except Exception as exc:
            notes.append(
                f"config.json unavailable ({_brief(exc)}); KV cache will be approximated"
            )
            out["config"] = {}

        out["quantization"] = detect_quantization(spec.repo_id, config=out.get("config"))
        if getattr(info, "gated", None):
            notes.append(
                f"repo is gated ({info.gated}); the runtime needs an accepted licence and a token"
            )
        return out

    def _from_cards(self, spec: ModelSpec, notes: list[str]) -> dict[str, Any]:
        """Size from the bundled model cards, which work offline."""
        out: dict[str, Any] = {"config": {}, "is_gguf": False}
        out["quantization"] = detect_quantization(spec.repo_id)
        try:
            from ..model_cards import get_model_card_registry

            registry = get_model_card_registry()
            model = spec.repo_id.rsplit("/", 1)[-1]
            card = None
            for provider in ("huggingface", "ollama", "deepinfra"):
                card = registry.get(provider, model) or registry.get(provider, spec.repo_id)
                if card is not None:
                    break
            if card is not None:
                window = card.get_context_length()
                if window:
                    out["config"] = {"max_position_embeddings": int(window)}
                    notes.append(f"context window {window:,} from the local model card")
                params = getattr(getattr(card, "architecture", None), "parameters_b", None)
                if params:
                    out["params"] = int(float(params) * 1e9)
                    notes.append(f"parameter count {params}B from the local model card")
        except Exception:
            logger.debug("model-card sizing fallback failed", exc_info=True)

        if "params" not in out:
            params = _params_from_name(spec.repo_id)
            if params:
                out["params"] = params
                notes.append(
                    f"parameter count {params / 1e9:.0f}B inferred from the repo name -- a guess, "
                    f"not metadata"
                )
        return out

    # -- arithmetic -------------------------------------------------------

    def _weights_gb(
        self, metadata: dict[str, Any], quantization: Quantization, notes: list[str]
    ) -> float:
        params = metadata.get("params")
        if not params:
            notes.append(
                "parameter count unknown; assuming 7B, which is a guess and may be very wrong"
            )
            params = 7_000_000_000

        if quantization is Quantization.NONE and metadata.get("params_by_dtype"):
            # Honour the dtypes the repo actually stores rather than assuming
            # 2 bytes: an fp32 checkpoint is twice the size people expect.
            total = sum(
                count * _DTYPE_BYTES.get(dtype.upper(), 2.0)
                for dtype, count in metadata["params_by_dtype"].items()
            )
            gb = total / 1024**3
            notes.append(f"weights {gb:.1f} GB from the per-dtype parameter map")
            return gb

        per = bytes_per_param(quantization)
        gb = params * per / 1024**3
        notes.append(
            f"weights {gb:.1f} GB = {params:,} params x {per} bytes ({quantization})"
        )
        return gb

    def _resolve_context(
        self, spec: ModelSpec, metadata: dict[str, Any], notes: list[str]
    ) -> int:
        model_max = int((metadata.get("config") or {}).get("max_position_embeddings") or 0)
        context = spec.context_length
        if model_max and context > model_max:
            notes.append(
                f"requested context {context:,} exceeds the model's {model_max:,}; using the model's"
            )
            context = model_max
        return context

    def _kv_gb(
        self,
        metadata: dict[str, Any],
        context: int,
        quantization: Quantization,
        notes: list[str],
    ) -> float:
        config = metadata.get("config") or {}
        layers = int(config.get("num_hidden_layers") or config.get("n_layer") or 0)
        hidden = int(config.get("hidden_size") or config.get("n_embd") or 0)
        heads = int(config.get("num_attention_heads") or config.get("n_head") or 0)
        # The GQA term. Defaulting to `heads` when absent is correct -- a model
        # without this key is doing full multi-head attention -- but reading
        # `heads` when the key *is* present would overestimate Llama-3.3-70B's
        # cache by 8x, so it is read explicitly.
        kv_heads = int(config.get("num_key_value_heads") or heads or 0)
        head_dim = int(config.get("head_dim") or 0) or (
            hidden // heads if hidden and heads else 0
        )

        if not (layers and kv_heads and head_dim):
            # No architecture metadata -- a gated repo whose config.json is not
            # readable, typically. Scale from the parameter count instead.
            #
            # The constant is deliberately **conservative**: measured against
            # models where the real figures are available, bytes-per-token per
            # billion parameters runs about 3.2 (Qwen3-30B), 4.6
            # (Llama-3.3-70B) and 7.5 (Qwen2.5-7B) KB. Taking the high end
            # overestimates large models by up to ~2.5x, and that is the right
            # direction to be wrong in: overestimating buys a bigger GPU, while
            # underestimating OOMs on the VM *after* billing has started.
            per_token_per_b = 8 * 1024  # bytes
            params_b = max(metadata.get("params", 7_000_000_000), 1) / 1e9
            approx = per_token_per_b * params_b * context / 1024**3
            approx = max(0.25, min(approx, 200.0))
            notes.append(
                f"KV cache ~{approx:.1f} GB -- approximated from {params_b:.0f}B params at "
                f"8 KB/token/B, because the repo exposes no layer/head configuration. "
                f"Deliberately conservative; the real figure is usually lower"
            )
            return approx

        # KV cache entries are stored at the cache dtype, which vLLM keeps at
        # 16-bit even for a 4-bit weight quantization unless explicitly told
        # otherwise. Assuming the weight dtype here would badly undercount.
        kv_bytes = 1.0 if quantization in (Quantization.FP8, Quantization.INT8) else 2.0
        total = 2 * layers * kv_heads * head_dim * kv_bytes * context
        gb = total / 1024**3
        notes.append(
            f"KV cache {gb:.1f} GB = 2 x {layers} layers x {kv_heads} kv-heads x "
            f"{head_dim} head-dim x {kv_bytes:g} bytes x {context:,} ctx"
        )
        if heads and kv_heads and heads != kv_heads:
            notes.append(
                f"grouped-query attention: {heads} query heads share {kv_heads} kv heads, "
                f"so the cache is {heads / kv_heads:.0f}x smaller than it looks"
            )
        return gb

    def _choose_sku(
        self,
        metadata: dict[str, Any],
        quantization: Quantization,
        weights_gb: float,
        context: int,
        required: float,
        notes: list[str],
    ) -> tuple[str, float, bool, int, float, float]:
        """Walk the ladder, shrinking context before giving up.

        Returns ``(sku, usable_gb, fits, context, kv_gb, required)``.

        Context is reduced before declaring a model unservable, because a 30B
        model at 8k context on an L4 is a useful thing and refusing it on the
        grounds that 262k does not fit would be unhelpfully literal.
        """
        candidates: list[str] = []
        for configured in self.ladder:
            resolved = self._resolve(configured)
            if resolved is None:
                notes.append(f"ignoring unknown SKU {configured!r} in the configured ladder")
            elif resolved not in candidates:
                candidates.append(resolved)
        if not candidates:
            # Falling back to Colab's ladder inside another vendor's catalogue
            # would size against GPUs that vendor does not sell, so there the
            # fallback is the catalogue itself, cheapest first.
            if self.skus is GPU_SKUS:
                notes.append(
                    f"no known SKUs in the configured ladder {self.ladder}; "
                    f"using the default ladder"
                )
                candidates = list(DEFAULT_SKU_LADDER)
            else:
                notes.append(
                    f"no known SKUs in the configured ladder {self.ladder}; "
                    f"using the backend catalogue cheapest-first"
                )
                candidates = sorted(
                    self.skus,
                    key=lambda n: (
                        self.skus[n].cost_per_hour
                        if self.skus[n].cost_per_hour is not None
                        else float("inf"),
                        self.skus[n].vram_gb,
                    ),
                )
            if not candidates:
                raise ValueError(
                    "The sizer was given an empty SKU catalogue, so there is "
                    "nothing to size against."
                )

        kv_gb = required - weights_gb - self.overhead_gb
        for name in candidates:
            usable = self.skus[name].vram_gb * self.gpu_memory_utilization
            budget = usable * (1 - self.headroom_fraction)
            if required <= budget:
                usable = budget
                notes.append(
                    f"{name}: {self.skus[name].vram_gb:.0f} GB x "
                    f"{self.gpu_memory_utilization:g} util x "
                    f"{1 - self.headroom_fraction:g} headroom = {budget:.1f} GB usable -- fits"
                )
                if self.skus[name].notes:
                    notes.append(f"{name}: {self.skus[name].notes}")
                return name, usable, True, context, kv_gb, required

            # Try a smaller context on this SKU before moving up a rung: a
            # bigger GPU costs money, a shorter context costs nothing.
            shrunk = self._fit_context(metadata, quantization, weights_gb, budget, context)
            if shrunk is not None:
                new_context, new_kv = shrunk
                usable = budget
                new_required = weights_gb + new_kv + self.overhead_gb
                notes.append(
                    f"{name}: {budget:.1f} GB usable -- fits at {new_context:,} ctx instead of "
                    f"{context:,} (KV {new_kv:.1f} GB)"
                )
                if self.skus[name].notes:
                    notes.append(f"{name}: {self.skus[name].notes}")
                return name, usable, True, new_context, new_kv, new_required
            notes.append(f"{name}: {budget:.1f} GB usable -- needs {required:.1f} GB, skipping")

        largest = candidates[-1]
        # The same budget the loop compared against, not the sticker VRAM -- a
        # refusal that quotes a bigger number than the decision used invites
        # the reader to conclude the sizer is being pessimistic.
        budget = (
            self.skus[largest].vram_gb * self.gpu_memory_utilization
            * (1 - self.headroom_fraction)
        )
        notes.append(
            f"nothing on the ladder fits: largest is {largest} at {budget:.1f} GB usable, "
            f"estimate is {required:.1f} GB"
        )
        return largest, budget, False, context, kv_gb, required

    def _resolve(self, name: str) -> str | None:
        """Map a configured spelling to a name in this sizer's catalogue."""
        key = str(name).strip()
        if key in self.skus:
            return key
        resolved = self.aliases.get(key.lower())
        return resolved if resolved in self.skus else None

    def _fit_context(
        self,
        metadata: dict[str, Any],
        quantization: Quantization,
        weights_gb: float,
        budget: float,
        context: int,
    ) -> tuple[int, float] | None:
        """Largest power-of-two-ish context that fits *budget*, or ``None``.

        Floors at 4096: below that a served model is not much use, and silently
        handing someone a 1k window would be a worse outcome than saying it does
        not fit.
        """
        spare = budget - weights_gb - self.overhead_gb
        if spare <= 0:
            return None
        for candidate in (131072, 65536, 32768, 16384, 8192, 4096):
            if candidate >= context:
                continue
            kv = self._kv_gb(metadata, candidate, quantization, [])
            if kv <= spare:
                return candidate, kv
        return None

    def _refusal_advice(
        self, metadata: dict[str, Any], weights_gb: float, quantization: Quantization
    ) -> tuple[str, ...]:
        """Say what *would* work, rather than only that this does not.

        A refusal with no alternative sends the caller to guess, and guessing
        here is expensive: the next guess is also a launch.
        """
        advice: list[str] = []
        if quantization is Quantization.NONE:
            quarter = weights_gb / 4
            advice.append(
                f"try a 4-bit build (AWQ/GPTQ): weights would be ~{quarter:.1f} GB instead of "
                f"{weights_gb:.1f} GB, which is usually the difference between fitting and not"
            )
        params = metadata.get("params")
        if params:
            billions = params / 1e9
            smaller = next((size for size in (70, 32, 14, 8, 7, 4, 3) if size < billions * 0.6), None)
            if smaller:
                advice.append(
                    f"or a smaller model in the same family -- around {smaller}B would fit the "
                    f"ladder comfortably"
                )
        advice.append(
            "or serve it somewhere with more VRAM; llmcore will not provision compute it has "
            "already calculated cannot hold the model"
        )
        return tuple(advice)


def _brief(exc: BaseException) -> str:
    """One readable line from an exception.

    Hub errors carry multi-line bodies with request ids and a paragraph of
    guidance. Pasting that into a sizing note makes the arithmetic -- the point
    of the notes -- unreadable.
    """
    import re as _re

    text = " ".join(str(exc).split())
    # Hub errors carry a request id that is noise in a sizing note.
    text = _re.sub(r"\(Request ID: [^)]*\)\s*", "", text)
    return text if len(text) <= 140 else text[:137] + "..."


def _with_context(spec: ModelSpec, context: int) -> ModelSpec:
    from dataclasses import replace

    return replace(spec, context_length=context)


def _params_from_name(repo_id: str) -> int | None:
    """Pull a parameter count out of a repo name (``...-7B-...``).

    A last resort, and labelled as a guess wherever it is used. It is still
    better than assuming every model is 7B, because the names are right far
    more often than not.
    """
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", re.sub(r"[-_]", " ", repo_id))
    if not match:
        return None
    try:
        return int(float(match.group(1)) * 1e9)
    except ValueError:
        return None

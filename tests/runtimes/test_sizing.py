# tests/runtimes/test_sizing.py
"""Sizing (spec phase R2).

Sizing decides whether to spend money and how much, so the tests care about two
things above all:

* **the direction of every error** — overestimating VRAM buys a bigger GPU,
  underestimating OOMs on the VM *after* billing has started, so unknowns must
  round toward "need more";
* **the arithmetic being inspectable** — a sizer that answers "use an A100" and
  shows nothing is impossible to argue with.

The Hub is stubbed. One test hits it for real and is marked, because the thing
most likely to break here is the Hub's metadata shape rather than the maths.
"""

from __future__ import annotations

import pytest

from llmcore.runtimes.models import ModelSpec, Quantization
from llmcore.runtimes.sizing import (
    DEFAULT_SKU_LADDER,
    GPU_SKUS,
    Sizer,
    bytes_per_param,
    detect_quantization,
    resolve_sku,
)

# Real numbers from Qwen2.5-7B-Instruct's config.json.
QWEN7B = {
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "hidden_size": 3584,
    "max_position_embeddings": 32768,
    "torch_dtype": "bfloat16",
}

# Llama-3.3-70B: the model where reading the wrong head count is an 8x error.
LLAMA70B = {
    "num_hidden_layers": 80,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "hidden_size": 8192,
    "max_position_embeddings": 131072,
}


def sizer(**kwargs) -> Sizer:
    kwargs.setdefault("offline", True)
    return Sizer(**kwargs)


def stub(monkeypatch, s: Sizer, *, params: int, config: dict, is_gguf: bool = False,
         quantization=None, params_by_dtype=None) -> None:
    """Make the sizer's metadata step return known values."""

    def _metadata(spec, notes):
        out = {
            "repo_id": spec.repo_id,
            "params": params,
            "config": config,
            "is_gguf": is_gguf,
            "quantization": quantization if quantization is not None
            else detect_quantization(spec.repo_id, config=config),
        }
        if params_by_dtype:
            out["params_by_dtype"] = params_by_dtype
        return out

    monkeypatch.setattr(s, "_metadata", _metadata)


# ---------------------------------------------------------------------------
# SKU table
# ---------------------------------------------------------------------------


class TestSkuTable:
    def test_every_sku_name_is_one_the_cli_accepts(self):
        """The draft spec listed A100-40 and A100-80 as rungs; `colab new`
        accepts only T4, L4, G4, H100, A100, so a plan naming A100-80 could
        never be provisioned."""
        accepted = {"T4", "L4", "G4", "H100", "A100"}
        assert {sku.gpu_flag for sku in GPU_SKUS.values()} <= accepted

    def test_the_draft_spellings_still_resolve(self):
        assert resolve_sku("A100-40") == "A100"
        assert resolve_sku("a100_80") == "A100"

    def test_an_unknown_spelling_resolves_to_nothing(self):
        assert resolve_sku("B200") is None

    def test_the_ladder_is_cheapest_first(self):
        vram = [GPU_SKUS[name].vram_gb for name in DEFAULT_SKU_LADDER]
        assert vram == sorted(vram)

    def test_a100_is_sized_conservatively(self):
        """Colab may hand out a 40 GB or an 80 GB card and the CLI cannot ask.
        Sizing for 40 means a plan is never short."""
        assert GPU_SKUS["A100"].vram_gb == 40.0


# ---------------------------------------------------------------------------
# Quantization detection
# ---------------------------------------------------------------------------


class TestQuantizationDetection:
    @pytest.mark.parametrize(
        ("repo", "expected"),
        [
            ("Qwen/Qwen2.5-7B-Instruct-AWQ", Quantization.AWQ),
            ("TheBloke/Llama-2-7B-GPTQ", Quantization.GPTQ),
            ("x/model-Q4_K_M", Quantization.GGUF_Q4),
            ("x/model-IQ4_XS", Quantization.GGUF_Q4),
            ("x/model-Q8_0", Quantization.GGUF_Q8),
            ("x/model-fp8", Quantization.FP8),
            ("meta-llama/Llama-3.3-70B-Instruct", None),
        ],
    )
    def test_from_the_name(self, repo, expected):
        assert detect_quantization(repo) is expected

    def test_the_owner_name_is_not_searched(self):
        """An organisation called 'awq-labs' does not make everything it
        publishes quantized."""
        assert detect_quantization("awq-labs/plain-model") is None

    def test_config_beats_the_name(self):
        """A repo declaring quantization_config is authoritative about itself;
        a name is a convention."""
        assert (
            detect_quantization(
                "x/model-AWQ", config={"quantization_config": {"quant_method": "gptq"}}
            )
            is Quantization.GPTQ
        )

    def test_bitsandbytes_bit_width_is_read(self):
        assert (
            detect_quantization("x/m", config={"quantization_config": {"quant_method": "bnb", "bits": 4}})
            is Quantization.INT4
        )

    def test_gguf_levels_are_distinct(self):
        """Q4 and Q8 differ by 2x in weight bytes, which is routinely the
        difference between fitting a 24 GB card and not."""
        assert bytes_per_param(Quantization.GGUF_Q8) > bytes_per_param(Quantization.GGUF_Q4) * 1.5


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------


class TestWeights:
    def test_weights_come_from_the_parameter_count(self, monkeypatch):
        s = sizer()
        stub(monkeypatch, s, params=7_000_000_000, config=QWEN7B)
        plan = s.estimate_sync(ModelSpec(repo_id="x/qwen-7b", context_length=8192))
        assert any("weights 13.0 GB" in note for note in plan.notes)

    def test_a_per_dtype_map_is_honoured_over_assuming_two_bytes(self, monkeypatch):
        """An fp32 checkpoint is twice the size people expect."""
        s = sizer()
        stub(
            monkeypatch, s, params=1_000_000_000, config=QWEN7B,
            params_by_dtype={"F32": 1_000_000_000},
        )
        plan = s.estimate_sync(ModelSpec(repo_id="x/fp32-1b", context_length=4096))
        assert any("per-dtype parameter map" in note for note in plan.notes)
        assert plan.vram_required_gb > 3.7   # 1B × 4 bytes

    def test_a_four_bit_build_is_a_quarter_of_the_weights(self, monkeypatch):
        s = sizer()
        stub(monkeypatch, s, params=70_000_000_000, config=LLAMA70B)
        full = s.estimate_sync(ModelSpec(repo_id="x/l70", context_length=8192))
        awq = s.estimate_sync(
            ModelSpec(repo_id="x/l70", context_length=8192, quantization=Quantization.AWQ)
        )
        assert awq.vram_required_gb < full.vram_required_gb / 3

    def test_an_unknown_parameter_count_says_it_is_guessing(self, monkeypatch):
        s = sizer()
        stub(monkeypatch, s, params=0, config={})
        plan = s.estimate_sync(ModelSpec(repo_id="x/mystery", context_length=4096))
        assert any("guess" in note for note in plan.notes)


class TestKvCache:
    def test_grouped_query_attention_uses_the_kv_head_count(self, monkeypatch):
        """Reading num_attention_heads instead of num_key_value_heads
        overestimates Llama-3.3-70B's cache by 8x -- the single biggest sizing
        error available."""
        s = sizer()
        stub(monkeypatch, s, params=70_000_000_000, config=LLAMA70B)
        plan = s.estimate_sync(ModelSpec(repo_id="x/l70", context_length=8192))
        kv_note = next(note for note in plan.notes if note.startswith("KV cache"))
        assert "8 kv-heads" in kv_note
        assert any("64 query heads share 8 kv heads" in note for note in plan.notes)

    def test_the_kv_arithmetic_is_shown(self, monkeypatch):
        s = sizer()
        stub(monkeypatch, s, params=7_000_000_000, config=QWEN7B)
        plan = s.estimate_sync(ModelSpec(repo_id="x/q7", context_length=32768))
        assert any(
            "2 × 28 layers × 4 kv-heads × 128 head-dim" in note for note in plan.notes
        )

    def test_kv_scales_with_context(self, monkeypatch):
        s = sizer()
        stub(monkeypatch, s, params=7_000_000_000, config=QWEN7B)
        small = s.estimate_sync(ModelSpec(repo_id="x/q7", context_length=4096))
        big = s.estimate_sync(ModelSpec(repo_id="x/q7", context_length=32768))
        assert big.vram_required_gb > small.vram_required_gb

    def test_the_cache_stays_16_bit_under_four_bit_weights(self, monkeypatch):
        """vLLM keeps the KV cache at 16-bit even for a 4-bit weight
        quantization, so assuming the weight dtype would badly undercount."""
        s = sizer()
        stub(monkeypatch, s, params=7_000_000_000, config=QWEN7B)
        none = s.estimate_sync(ModelSpec(repo_id="x/q7", context_length=32768))
        awq = s.estimate_sync(
            ModelSpec(repo_id="x/q7", context_length=32768, quantization=Quantization.AWQ)
        )
        kv_none = next(n for n in none.notes if n.startswith("KV cache"))
        kv_awq = next(n for n in awq.notes if n.startswith("KV cache"))
        assert "2 bytes" in kv_none and "2 bytes" in kv_awq

    def test_the_fallback_errs_toward_needing_more(self, monkeypatch):
        """No architecture metadata -- a gated repo, typically. The estimate
        must not come out absurdly small, because that OOMs after billing
        starts."""
        s = sizer()
        stub(monkeypatch, s, params=70_000_000_000, config={})
        plan = s.estimate_sync(ModelSpec(repo_id="x/gated-70b", context_length=8192))
        kv_note = next(note for note in plan.notes if note.startswith("KV cache"))
        assert "approximated" in kv_note and "conservative" in kv_note
        # The real figure for a 70B at 8k is ~2.7 GB; the approximation must be
        # in that neighbourhood or above, never near zero.
        assert plan.vram_required_gb > 131.0


class TestContext:
    def test_a_request_above_the_models_window_is_clamped(self, monkeypatch):
        s = sizer()
        stub(monkeypatch, s, params=7_000_000_000, config=QWEN7B)   # 32k model
        plan = s.estimate_sync(ModelSpec(repo_id="x/q7", context_length=200_000))
        assert plan.context_length <= 32768
        assert any("exceeds the model's" in note for note in plan.notes)

    def test_context_shrinks_before_a_bigger_gpu_is_bought(self, monkeypatch):
        """A bigger GPU costs money; a shorter context costs nothing.

        The real Qwen2.5-7B parameter count is used here, because this is the
        case observed against the live Hub: 14.2 GB of weights plus 1.8 GB of
        KV plus overhead is 18.5 GB, just over an L4's 18.4 GB budget. Rounding
        the count down to 7B makes it fit at full context and the test stops
        testing anything.
        """
        s = sizer()
        stub(monkeypatch, s, params=7_615_616_512, config=QWEN7B)
        plan = s.estimate_sync(ModelSpec(repo_id="x/q7", context_length=32768))
        assert plan.sku == "L4"
        assert plan.context_length == 16384
        assert any("fits at 16,384 ctx instead of 32,768" in note for note in plan.notes)

    def test_the_plan_reports_the_context_it_was_sized_for(self, monkeypatch):
        s = sizer()
        stub(monkeypatch, s, params=7_615_616_512, config=QWEN7B)
        plan = s.estimate_sync(ModelSpec(repo_id="x/q7", context_length=32768))
        assert plan.spec.context_length == plan.context_length

    def test_it_will_not_shrink_below_a_usable_window(self, monkeypatch):
        """Silently handing someone a 1k window is worse than saying it does
        not fit."""
        s = sizer(ladder=("T4",))
        stub(monkeypatch, s, params=30_000_000_000, config=QWEN7B)
        plan = s.estimate_sync(ModelSpec(repo_id="x/big", context_length=32768))
        assert plan.fits is False


class TestSkuChoice:
    def test_the_cheapest_fitting_sku_wins(self, monkeypatch):
        s = sizer()
        stub(monkeypatch, s, params=3_000_000_000, config=QWEN7B)
        plan = s.estimate_sync(ModelSpec(repo_id="x/small", context_length=4096))
        assert plan.sku == "T4"

    def test_headroom_is_reserved(self, monkeypatch):
        """An estimate that exactly fills a card OOMs on the first long
        request, after the money has started."""
        s = sizer(headroom_fraction=0.15)
        stub(monkeypatch, s, params=7_000_000_000, config=QWEN7B)
        plan = s.estimate_sync(ModelSpec(repo_id="x/q7", context_length=4096))
        assert plan.headroom_gb >= 0

    def test_reported_capacity_is_what_the_decision_used(self, monkeypatch):
        """Reporting sticker VRAM would make a plan look like it had several GB
        more room than the sizer believed."""
        s = sizer(headroom_fraction=0.15, gpu_memory_utilization=0.90)
        stub(monkeypatch, s, params=3_000_000_000, config=QWEN7B)
        plan = s.estimate_sync(ModelSpec(repo_id="x/small", context_length=4096))
        assert plan.vram_available_gb == pytest.approx(16.0 * 0.90 * 0.85, abs=0.01)

    def test_a_restricted_ladder_is_honoured(self, monkeypatch):
        """An account that cannot get A100s should not be told twice."""
        s = sizer(ladder=("T4", "L4"))
        stub(monkeypatch, s, params=70_000_000_000, config=LLAMA70B)
        plan = s.estimate_sync(ModelSpec(repo_id="x/l70", context_length=8192))
        assert plan.fits is False and plan.sku in ("T4", "L4")

    def test_an_unknown_sku_in_the_ladder_is_skipped_with_a_note(self, monkeypatch):
        s = sizer(ladder=("B200", "L4"))
        stub(monkeypatch, s, params=3_000_000_000, config=QWEN7B)
        plan = s.estimate_sync(ModelSpec(repo_id="x/small", context_length=4096))
        assert plan.sku == "L4"
        assert any("unknown SKU" in note for note in plan.notes)

    def test_every_sku_considered_is_explained(self, monkeypatch):
        s = sizer()
        stub(monkeypatch, s, params=30_000_000_000, config=QWEN7B)
        plan = s.estimate_sync(ModelSpec(repo_id="x/30b", context_length=8192))
        assert sum(1 for note in plan.notes if note.startswith(("T4:", "L4:", "G4:", "A100:"))) >= 3

    def test_a_burn_rate_is_reported(self, monkeypatch):
        s = sizer()
        stub(monkeypatch, s, params=3_000_000_000, config=QWEN7B)
        plan = s.estimate_sync(ModelSpec(repo_id="x/small", context_length=4096))
        assert plan.estimated_cost_per_hour == GPU_SKUS["T4"].compute_units_per_hour


class TestRefusal:
    def test_nothing_fitting_is_a_refusal_not_a_plan(self, monkeypatch):
        s = sizer()
        stub(monkeypatch, s, params=400_000_000_000, config=LLAMA70B)
        plan = s.estimate_sync(ModelSpec(repo_id="x/enormous", context_length=8192))
        assert plan.fits is False

    def test_a_refusal_suggests_something_concrete(self, monkeypatch):
        """A refusal with no alternative sends the caller to guess, and the
        next guess is also a launch."""
        s = sizer()
        stub(monkeypatch, s, params=400_000_000_000, config=LLAMA70B)
        plan = s.estimate_sync(ModelSpec(repo_id="x/enormous", context_length=8192))
        joined = " ".join(plan.notes)
        assert "4-bit" in joined and "smaller model" in joined

    def test_a_refusal_quotes_the_budget_the_decision_used(self, monkeypatch):
        s = sizer()
        stub(monkeypatch, s, params=400_000_000_000, config=LLAMA70B)
        plan = s.estimate_sync(ModelSpec(repo_id="x/enormous", context_length=8192))
        refusal = next(note for note in plan.notes if "nothing on the ladder" in note)
        assert f"{plan.vram_available_gb:.1f} GB usable" in refusal


class TestRecipe:
    def test_a_gguf_repo_switches_recipe(self, monkeypatch):
        s = sizer()
        stub(monkeypatch, s, params=7_000_000_000, config={}, is_gguf=True)
        plan = s.estimate_sync(ModelSpec(repo_id="x/model-GGUF", context_length=4096))
        assert plan.recipe == "llamacpp"
        assert any("GGUF" in note for note in plan.notes)

    def test_a_safetensors_repo_uses_vllm(self, monkeypatch):
        s = sizer()
        stub(monkeypatch, s, params=7_000_000_000, config=QWEN7B)
        assert s.estimate_sync(ModelSpec(repo_id="x/q7", context_length=4096)).recipe == "vllm"


class TestOffline:
    def test_offline_sizing_never_touches_the_network(self, monkeypatch):
        def explode(*a, **k):
            raise AssertionError("the Hub must not be contacted when offline=True")

        monkeypatch.setattr("huggingface_hub.HfApi.model_info", explode)
        plan = sizer(offline=True).estimate_sync(
            ModelSpec(repo_id="Qwen/Qwen2.5-7B-Instruct", context_length=8192)
        )
        assert plan.sku in GPU_SKUS

    def test_a_parameter_count_is_read_from_the_repo_name(self):
        plan = sizer(offline=True).estimate_sync(
            ModelSpec(repo_id="someone/mystery-13B-chat", context_length=4096)
        )
        assert any("13B" in note or "13" in note for note in plan.notes)


class TestEstimateIsFree:
    @pytest.mark.asyncio
    async def test_the_async_wrapper_agrees_with_the_sync_one(self, monkeypatch):
        s = sizer()
        stub(monkeypatch, s, params=7_000_000_000, config=QWEN7B)
        spec = ModelSpec(repo_id="x/q7", context_length=8192)
        assert (await s.estimate(spec)).sku == s.estimate_sync(spec).sku

    def test_sizing_a_model_provisions_nothing(self, monkeypatch):
        """The Sizer has no handle on a backend at all, which is the structural
        guarantee rather than a promise."""
        s = sizer()
        assert not any(
            hasattr(s, attribute) for attribute in ("up", "down", "_cli", "_run")
        )


@pytest.mark.live
class TestAgainstTheRealHub:
    """The Hub's metadata shape is the thing most likely to change under us."""

    @pytest.mark.asyncio
    async def test_a_known_model_sizes_from_real_metadata(self):
        plan = await Sizer().estimate(
            ModelSpec(repo_id="Qwen/Qwen2.5-7B-Instruct", context_length=8192)
        )
        assert plan.fits
        # 7.6B at bf16 is ~14.2 GB of weights; anything far from that means the
        # parameter map was not read.
        assert 13.0 < plan.vram_required_gb < 20.0
        assert any("safetensors index" in note for note in plan.notes)

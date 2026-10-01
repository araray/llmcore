# Remote GPU runtimes — usage guide

Size an open-weights model, provision compute, serve it, and reach it through
the same `llm.chat()` as any hosted API.

Design and safety rationale: [`COLAB_RUNTIME_SPEC.md`](COLAB_RUNTIME_SPEC.md).
This document is how to use it.

---

## Read this first

**A runtime bills per minute from the moment compute is assigned, whether or
not anything calls it.** That makes this subsystem unlike every other provider
in llmcore, and the defaults are built around it:

- it is **off** until `[runtimes] enabled = true`;
- `LLMCore.create()` contacts no backend regardless of config;
- `up()` refuses without explicit spend confirmation;
- `estimate()` is free, read-only, and works while the subsystem is disabled —
  deciding whether to spend should not require enabling spend;
- state is written **before** provisioning, as inspectable JSON under
  `~/.llmcore/runtimes`, so a runtime can always be found and killed;
- idle **and** hard deadlines are set at creation, because an idle reaper does
  not stop a runtime that is busy in a loop;
- `close()` detaches but does **not** tear down. A process exiting is not a
  reason to destroy compute you are paying for. `down_all()` is explicit.

If you only remember one command: `llmcore-runtimes status` tells you what is
running, including things llmcore did not start.

---

## Install

```bash
pip install "llmcore[runtimes]"      # or: pip install google-colab-cli
colab --help                         # the official Colab CLI must authenticate once
```

```toml
[runtimes]
enabled = true
default_backend = "colab"

  [runtimes.defaults]
  confirm_spend = true
  idle_minutes = 45
  max_lifetime_minutes = 240
```

---

## 1. Size it first (free)

```bash
llmcore-runtimes estimate Qwen/Qwen2.5-7B-Instruct --context 16384
```

```
model      Qwen/Qwen2.5-7B-Instruct
recipe     vllm
quant      none
context    16,384
needs      17.6 GB
sku        L4 (18.4 GB usable)
fits       yes
burn rate  ~4.82 compute units/hour while assigned

working:
  - parameters from the Hub safetensors index: 7,615,616,512 x BF16
  - weights 14.2 GB from the per-dtype parameter map
  - KV cache 0.9 GB = 2 x 28 layers x 4 kv-heads x 128 head-dim x 2 bytes x 16,384 ctx
  - grouped-query attention: 28 query heads share 4 kv heads, so the cache is 7x smaller than it looks
  - total = weights 14.2 + KV 0.9 + overhead 2.5 = 17.6 GB
  - T4: 12.2 GB usable -- needs 17.6 GB, skipping
  - L4: 24 GB x 0.9 util x 0.85 headroom = 18.4 GB usable -- fits
  - colab new --gpu L4
```

```python
plan = await llm.runtimes.estimate("Qwen/Qwen2.5-7B-Instruct", context_length=16384)
print(plan.sku, plan.fits, plan.vram_required_gb)
for note in plan.notes:
    print(" ", note)
```

The arithmetic is printed on purpose. A sizer that answers "use an A100" and
shows nothing is impossible to argue with, and the first question anyone has is
why a 30B model needs more than 24 GB.

**Every unknown rounds toward needing more.** Overestimating buys a bigger GPU;
underestimating OOMs on the VM after billing has started.

### When it does not fit

The sizer refuses with something concrete rather than just "no":

```
fits       NO
  - try a 4-bit build (AWQ/GPTQ): weights would be ~32.9 GB instead of 131.4 GB
  - or a smaller model in the same family -- around 32B would fit the ladder comfortably
  - or serve it somewhere with more VRAM
```

Quantization is detected from the repo's `quantization_config` or its name, and
can be forced:

```bash
llmcore-runtimes estimate meta-llama/Llama-3.3-70B-Instruct --quantization awq
```

Context is reduced before a bigger GPU is chosen — a bigger GPU costs money, a
shorter context costs nothing — with a floor at 4k.

---

## 2. Start it (this spends money)

```bash
llmcore-runtimes up Qwen/Qwen2.5-7B-Instruct --name q7 --context 16384 --yes
```

```python
handle = await llm.runtimes.up(
    "Qwen/Qwen2.5-7B-Instruct", name="q7", context_length=16384, confirm_spend=True
)
answer = await llm.chat("Explain GQA briefly.", provider_name="q7")
```

The endpoint is OpenAI-compatible, so no new provider class is involved: the
runtime is registered as a `vllm`-type instance under its own name and is then
an ordinary provider. That also means it works with routing:

```toml
[routing.pools.local_first]
targets = ["q7:Qwen/Qwen2.5-7B-Instruct", "openai:gpt-5.4?order=1"]
```

What happens, in order: size → create the session → **wait for it to appear**
before connecting to anything → mount Drive → restore or build the environment
→ fetch weights → start the server detached → tunnel it to localhost → wait for
`/v1/models` → attach.

**Any failure releases the VM before raising.** If the release *also* fails, the
log says `MAY STILL BE BILLING` in those words, because that is the one case you
have to act on yourself.

---

## 3. Watch it

```bash
llmcore-runtimes status
```

```
NAME             PHASE      SKU      MODEL                              EXPIRES
q7               ready      L4       Qwen/Qwen2.5-7B-Instruct           41m

! stray-session: orphan: a Colab session named 'stray-session' is running that
  llmcore has no record of. It may be billing. Adopt it to make it killable:
  llm.runtimes.adopt('stray-session', name='stray-session')
```

Orphan detection is a safety feature, not a nicety: an unknown running VM is
unmonitored spend. Adopting one is what makes it killable.

```bash
llmcore-runtimes adopt stray-session --name rescued
llmcore-runtimes down rescued
```

An adopted runtime is marked `DEGRADED` and says it does not know what it is
running, because it does not. Claiming otherwise would be worse than admitting
the gap.

### Liveness

A supervisor probes each runtime's endpoint on a timer and marks it `DEGRADED`
after **three** consecutive failures — one missed poll is usually the tunnel
reconnecting. A failed probe never tears a runtime down: a degraded runtime is
still assigned and still billing, and letting a transient network problem
destroy an expensive VM would be worse than the problem. Releasing it is the
deadlines' job, or yours.

```bash
llmcore-runtimes logs q7                      # the model server's own output
llmcore-runtimes logs q7 --component bootstrap
```

---

## 4. Stop paying

```bash
llmcore-runtimes down q7
llmcore-runtimes down --all
```

```python
await llm.runtimes.down("q7")        # unregister the provider, then release
await llm.runtimes.down_all()
```

`down` is idempotent — stopping something already gone is how recovery from a
half-failed start works, so it cannot raise.

`--detach` stops tracking **without** releasing, and says so loudly. The runtime
stays in the `DETACHED` phase, which still counts as billing.

---

## 5. Make the next start fast

A cold start resolves dependencies and downloads weights. Both are cached in
Drive, and both can be prepared on a **CPU** VM so no GPU minutes are spent on
`pip install`:

```bash
llmcore-runtimes bake --recipe vllm
llmcore-runtimes cache
```

Cached artefacts are gated on a sentinel written only after the artefact is
complete: a cache miss costs time, while a corrupt hit costs a debugging session
on a billing VM.

---

## Configuration

```toml
[runtimes]
enabled = false
default_backend = "colab"
state_dir = "~/.llmcore/runtimes"

  [runtimes.defaults]
  recipe = "vllm"
  confirm_spend = true
  idle_minutes = 45
  max_lifetime_minutes = 240
  supervise_seconds = 60          # 0 disables liveness AND the idle reaper
  gpu_memory_utilization = 0.90
  headroom_fraction = 0.15

  [runtimes.colab]
  cli_path = "colab"
  drive_cache_dir = "/content/drive/MyDrive/.llmcore-cache"
  sku_ladder = ["T4", "L4", "G4", "A100", "H100"]
  keepalive_seconds = 60
  # hf_token_env_var = "HF_TOKEN"
```

Trim `sku_ladder` to what your subscription can actually get: an account that
will never be given an A100 should not spend a minute discovering that on every
launch. These are the names `colab new --gpu` accepts; there is no way to ask
for a particular amount of A100 VRAM, so A100 is sized conservatively at 40 GB.
The older `A100-40`/`A100-80` spellings still resolve.

---

## What this does not claim

- **The end-to-end path has been run once against a real Colab T4** (2026-10-01,
  Qwen2.5-1.5B-Instruct on vLLM, reached through `llm.chat()` and through a
  pool, then released). That run found four bugs the whole test suite had
  passed, which is the honest argument for treating one live run as worth more
  than any amount of mocking here.

  What is still unproven: a single unattended `up()` with all four fixes
  applied — in the successful run the server was started by the corrected
  bootstrap invoked by hand, after the orchestration's own attempt had failed.
  Expect to hit something on a model or SKU combination nobody has tried.
- **A cold start takes tens of minutes**, most of it installing the serving
  stack. The measured run was 35 minutes to first token on a T4, with the pip
  cache misconfigured onto Google Drive (since fixed). `bake` exists precisely
  so this is paid once on a CPU VM.
- **Sizing is an estimate.** It is conservative by design, and a model that the
  sizer says fits can still fail to load for reasons no estimate can see
  (a custom kernel, an unexpected dtype, a vLLM version mismatch).
- **Gated repos need a token *and* an accepted licence.** The sizer reports a
  gated repo and falls back to approximating the KV cache, because it cannot
  read the config it would need.
- **The keepalive runs in the llmcore process**, so killing llmcore stops it.
  That is the safer default — a keepalive that outlives its owner is a leaked VM
  with a heartbeat — but it does mean a long run needs llmcore to stay up.
- **Colab's own session horizon still applies** and is surfaced as an ETA, never
  circumvented.

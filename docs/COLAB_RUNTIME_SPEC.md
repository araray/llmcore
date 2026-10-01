# Remote Compute Runtimes — Design & Specification (Colab first)

Let llmcore provision, control and serve models on remote GPU runtimes — Google
Colab first — so a model running on someone else's GPU is just another provider.

- **Status:** **R1 implemented** — `llmcore.runtimes` core, the safety model,
  and provider attachment, with a `FakeRuntime` for tests. **No real backend
  yet, so nothing can spend money.** R2 (sizing) onward not started — see §6.
- **Written:** 2026-09-29
- **Reference implementation studied:** `/av/repos/agent-lens`
  (`docs/colab-design.md`, 391 lines + ~3,750 lines across 13 modules) and
  `/av/repos/BellaVox` (the process agent-lens generalized)
- **Upstream CLI studied:** `/av/avalon/xrepos/google-colab-cli` (official)

---

## 1. Problem statement and the factoring decision

`agent-lens` already does this, and does it well: size a Hugging Face model,
pick a GPU SKU, boot a Colab VM, restore a cached venv + weights from Drive,
start vLLM on the VM, tunnel it to a local port over Colab's authenticated
WebSocket, **and register it as an llmcore provider**.

That last step is the tell. The capability is being built *on top of* llmcore by
a consumer, so every other llmcore consumer has to rebuild it. Meanwhile llmcore
already owns everything the endpoint needs once it exists: the `vllm` provider
speaks the OpenAI-compatible surface vLLM exposes, `ProviderManager` owns
registration and lifecycle, and the model-card registry owns capability
metadata.

**Decision: llmcore owns the runtime abstraction; agent-lens becomes a consumer
of it.** llmcore gains a `llmcore.runtimes` subsystem. `agent-lens colab` keeps
its CLI and its opinions (sizing heuristics, the "lens" adversarial plan review,
its dashboard) but delegates provisioning/bootstrap/tunnel/lifecycle to llmcore,
deleting most of its ~3,750 lines.

Generalize one level, not two: the abstraction is **"remote compute runtime"**,
with Colab as the first backend and RunPod / Modal / Lambda / plain SSH as
plausible later ones. The interface is designed for that, but only Colab is
implemented.

### Non-goals (v1)

Inherited from the reference design, and correct:

- No multi-tenant serving, queueing or cross-VM routing.
- No training or fine-tuning — inference only.
- No browser automation. We use the official CLI; if auth breaks, it fails loudly.
- No attempt to circumvent platform limits (no fake-activity tricks).

---

## 2. Safety model — this one is different from every other provider

Every other llmcore provider is stateless and costs money per request. **A Colab
runtime costs money per minute from the moment it is assigned, whether or not
anyone calls it.** The safety rules follow from that, and they are
non-negotiable design constraints rather than polish:

1. **No implicit spend.** Nothing provisions or *keeps* a runtime without an
   explicit caller action. `LLMCore.create()` must never boot a VM, no matter
   what is in the config. Read-only operations are read-only.
2. **No implicit persistence of spend.** A runtime that llmcore started is
   recorded in inspectable state so it can always be found and killed — a leaked
   VM is a leaked credit card.
3. **Bounded by default.** Idle reaping is **on** by default (45 min), because
   stopping saves money and the Drive cache makes restart cheap. Warn before the
   axe falls.
4. **Fail closed.** Any bootstrap failure releases the VM. Never leave one
   burning after an error. Keep the Drive cache.
5. **No implicit secrets.** HF tokens and Colab auth come from explicit config /
   existing CLI state; never in argv, never in logs, never persisted on the VM.

> **Improvement over the reference:** add a **spend ceiling**. A runtime carries
> `max_lifetime_minutes` (hard kill) and optionally `max_compute_units`. The
> reaper enforces both. The reference design has an idle reaper and notes
> Colab's ~12 h horizon but has no user-set hard cap; an idle reaper does not
> protect against a runtime that is *busy* in a loop.

---

## 3. Architecture

```
LLMCore
 ├── ProviderManager                       (existing)
 │     └── dynamically registered instance → the runtime's endpoint
 └── RuntimeManager                        (new)
       ├── ColabRuntime            backend: official colab CLI + SSH + tunnel
       ├── (future) SSHRuntime / RunPodRuntime / ModalRuntime
       ├── Sizer                   HF metadata → Plan (SKU, quant, ctx, VRAM)
       ├── ServerRecipe            vllm | llamacpp | tgi | custom
       ├── CacheStore              Drive (Colab) / volume (others): env + weights
       ├── Tunnel                  local port ⇄ remote 127.0.0.1:port
       ├── Keepalive + Reaper      liveness, idle kill, hard lifetime cap
       └── RuntimeState            inspectable JSON under ~/.llmcore/runtimes/
```

### 3.1 The runtime protocol

```python
@runtime_checkable
class ComputeRuntime(Protocol):
    name: str

    async def estimate(self, spec: ModelSpec) -> Plan: ...
    async def up(self, plan: Plan, *, name: str) -> RuntimeHandle: ...
    async def status(self, name: str | None = None) -> list[RuntimeStatus]: ...
    async def logs(self, name: str, *, component: str, tail: int) -> AsyncIterator[str]: ...
    async def down(self, name: str, *, release: bool = True) -> None: ...
    async def adopt(self, external_id: str, *, name: str) -> RuntimeHandle: ...
```

`RuntimeHandle` carries what the provider layer needs:

```python
@dataclass(slots=True)
class RuntimeHandle:
    name: str
    runtime: str                  # "colab"
    external_id: str              # colab session id
    base_url: str                 # http://127.0.0.1:<port>/v1
    served_model: str             # the HF repo id vLLM was told to serve
    api_style: str                # "openai"  → which llmcore provider to attach
    recipe: str                   # "vllm"
    sku: str                      # "L4" / "A100-40" / ...
    started_at: datetime
    idle_deadline: datetime | None
    hard_deadline: datetime | None
    state_path: Path
```

### 3.2 Provider attachment — the key integration

Once a handle exists, its endpoint is OpenAI-compatible, so **no new provider
class is needed**. `RuntimeManager.attach()` registers a `vllm`-type provider
instance into the live `ProviderManager` under the runtime's name:

```python
rt = await llm.runtimes.up("Qwen/Qwen3-30B-A3B-Instruct-2507", name="qwen30")
# -> provider instance "qwen30" now exists, type=vllm, base_url=the tunnel
answer = await llm.chat("Explain GQA briefly.", provider_name="qwen30")
await llm.runtimes.down("qwen30")          # unregisters, then releases the VM
```

This is why the abstraction is cheap: llmcore already has the client. Three
small additions are needed to `ProviderManager`:

- `register_instance(name, type, config)` / `unregister_instance(name)` —
  dynamic registration at runtime (today providers are only built in `__init__`).
- Instances marked ephemeral so `close_all()` tears down runtimes it owns.
- `get_provider()` raising a clear error when a runtime-backed instance exists
  but its runtime is `DEGRADED`.

> **Improvement:** because the handle records `api_style`, a future recipe that
> speaks a different protocol (TGI, llama.cpp server) attaches a different
> provider type without touching the runtime layer.

### 3.3 Sizing engine

Ported from `agent-lens/colab/sizing.py`, which is already well-specified:

1. **Metadata** — HF Hub API `safetensors` parameter map (exact params + dtypes);
   fall back to file sizes × dtypes; fall back to the local `model_cards`
   registry (offline).
2. **Weights** — Σ params × bytes/param, adjusted for quantization (explicit
   `--quant`, else detected from the repo name: `-AWQ`, `-GPTQ`, `IQ4_XS`; GGUF
   repos switch to the llama.cpp recipe).
3. **KV cache** — `2 × layers × kv_heads × head_dim × bytes × ctx`, GQA-aware,
   per architecture family. Default ctx = `min(model_max, 32k)`.
4. **Runtime overhead** — ~2–3 GB activation/fragmentation, target
   `gpu_memory_utilization = 0.90`.
5. **SKU ladder** — cheapest that fits with ≥15% headroom:
   `T4 16GB → L4 24GB → G4 24GB → A100 40GB → A100 80GB → H100`, pruning SKUs
   the account cannot get (a 400 from `colab new` prunes interactively). If it
   does not fit even quantized, **refuse with a concrete smaller suggestion**.
6. Every number is printed with its arithmetic. `estimate` is a pure dry run.

> **Improvement:** the sizer should write its `Plan` into the model card
> registry as a `runtime_hint` for that `(model, sku, quant, ctx)` tuple, so
> repeat launches skip the HF round trip and the ladder is learned rather than
> recomputed.

### 3.4 Bootstrap sequence (Colab)

Faithful to the reference, which is battle-tested:

```
1. estimate → Plan; print summary + burn rate; require explicit confirmation
2. colab new -s <name> --gpu <sku> [--high-mem]
3. guard: session appears in assignments before any SSH ("never ssh into the void")
4. ssh master up (ProxyCommand = colab ssh --proxy-mode; isolated ed25519 key;
   ControlMaster + ControlPersist)
5. push a versioned bootstrap bundle, then on the VM:
6.   mount Drive
7.   ensure pinned Python (runtime pin file in Drive)
8.   ensure env: tar-restore from Drive, else pip build with PIP_CACHE_DIR on
     Drive, then re-tar to Drive on a miss
9.   ensure weights: local? Drive? else snapshot_download → copy to Drive
     (sentinel files; allow_patterns = safetensors/config/tokenizer)
10.  setsid <recipe> serve --host 127.0.0.1 --port 8000 ... > server.log 2>&1 &
     ready marker once /v1/models answers
11. local: ssh -N -L 127.0.0.1:<free>:127.0.0.1:8000 over the master
12. keepalive on; write state; attach provider (§3.2)
```

Any failure → log pointer + **release the VM**, keep the Drive cache.
The HF token is forwarded over SSH **stdin** to the download step only.

A `bake` command pre-builds environment tars on a *CPU* VM so GPU minutes are
never spent on `pip install` — carried over from the reference and worth keeping.

### 3.5 Keepalive, liveness, reaping

- **Keepalive** — a kernel-side loop via `colab exec` stdin holds the kernel
  active, which holds the VM; PID tracked locally. The server itself runs
  `setsid`-detached so it survives kernel churn.
- **Liveness** — poll `GET /v1/models` through the tunnel every 60 s; three
  consecutive failures → `DEGRADED` with a restart hint. Restarting the *server*
  over the existing SSH master is cheap and safe; restarting the *VM* is not
  automatic.
- **Idle reaper** — vLLM exposes no last-request metric, so count traffic on the
  local tunnel side. After `idle_minutes` (default 45) → `down` + release. Warn
  10 minutes ahead in `status`.
- **Hard deadline** — `max_lifetime_minutes` (new, §2). Colab's own ~12 h
  horizon is surfaced as an ETA, never circumvented.

### 3.6 State

`~/.llmcore/runtimes/<name>.json` — small, human-readable, deletable:
the plan, the handle, PIDs (keepalive, tunnel, SSH master), timestamps,
deadlines, and the bootstrap log path. `status` reconciles state against
`colab ls`: sessions llmcore knows that are gone → mark stale; sessions that
exist but llmcore does not know → show as **orphans** with an `adopt` hint.
Orphan detection is a safety feature, not a nicety: an unknown running VM is
unmonitored spend.

---

## 4. Public API and CLI

```python
llm.runtimes.estimate("Qwen/Qwen3-30B-A3B-Instruct-2507", ctx=32768)
handle = await llm.runtimes.up(repo, name="qwen30", gpu="L4", idle_minutes=45)
await llm.runtimes.status()            # list[RuntimeStatus]
await llm.runtimes.logs("qwen30", component="server", follow=True)
await llm.runtimes.down("qwen30")      # unregister + release
await llm.runtimes.adopt("<session-id>", name="rescued")
```

CLI (`llmcore-runtimes`, mirroring the proven agent-lens surface):

```
llmcore-runtimes estimate <hf-repo> [--rev] [--ctx N] [--quant Q]
llmcore-runtimes up <hf-repo> [--name] [--gpu SKU] [--ctx N] [--quant]
                              [--idle-min N] [--max-lifetime-min N] [--recipe]
llmcore-runtimes status [NAME] [-f] | ps [--json] | logs [NAME] [--component]
llmcore-runtimes keepalive on|off [NAME]
llmcore-runtimes bake [--recipe vllm] | cache ls|gc
llmcore-runtimes down [NAME] [--all] | adopt <session-id> --name NAME
```

---

## 5. Configuration sketch

```toml
[runtimes]
enabled = true                   # gate the subsystem; NEVER auto-provisions
default_backend = "colab"
state_dir = "~/.llmcore/runtimes"

[runtimes.defaults]
recipe = "vllm"
idle_minutes = 45                # 0 disables the reaper
max_lifetime_minutes = 240       # hard kill regardless of activity
confirm_spend = true             # require explicit confirmation before `up`
gpu_memory_utilization = 0.90
headroom_fraction = 0.15

[runtimes.colab]
cli_path = "colab"               # auto-discovered; install hint if missing
drive_cache_dir = "/content/drive/MyDrive/.llmcore-cache"
sku_ladder = ["T4", "L4", "G4", "A100-40", "A100-80", "H100"]
# hf_token_env_var = "HF_TOKEN"
```

---

## 6. Implementation plan

| Phase | Scope | Gate |
|---|---|---|
| **R1** ✅ | `llmcore.runtimes` core: `ComputeRuntime` protocol, `ModelSpec`/`Plan`/`RuntimeHandle`/`RuntimeStatus`, `RuntimeStateStore`, config section, `RuntimeManager`, `llm.runtimes`, `FakeRuntime`. **No network.** | Landed 2026-09-30, 76 tests. Gate met: a runtime attaches as a real `VLLMProvider` instance at its tunnel URL, marked ephemeral, and detaches on `down()`. See §6.1 |
| **R2** ✅ | `Sizer` — HF metadata, KV math, quant detection, SKU ladder, `estimate`. GET-only, no spend. | **Met.** Verified against Qwen2.5-7B, Qwen3-30B (dense and AWQ), Llama-3.3-70B (gated, exercising the fallback) and gemma-3-4b on the live Hub. 45 tests |
| **R3** ⚠️ | `ColabRuntime` — CLI discovery, `new`, assignment guard, SSH master, bundle push, Drive cache, vLLM recipe, tunnel, ready marker. `up`/`down`/`status`/`logs`. | **Implemented, gate unmet.** Every path is tested against the real CLI's command surface, but "one real model served end-to-end" needs a real GPU VM and real compute units — that run belongs to whoever is paying. 58 tests |
| **R4** ✅ | Keepalive, liveness probe, idle reaper, hard deadline, orphan detection + `adopt`. | **Met.** `reap()` existed but nothing called it, so the deadlines were documentation; `up()` now starts a supervisor. Orphans are reported with the command that adopts them |
| **R5** ⚠️ | `bake`, Drive cache inventory + `cache gc`, `llamacpp`/GGUF recipe. | **Implemented, gate unmet** for the same reason as R3: "cold start is seconds" is a measurement on a real VM. `cache gc` is not implemented — `cache` lists, nothing deletes |
| **R6** ⚠️ | CLI + docs; **agent-lens migration guide** so it delegates here. | `llmcore-runtimes` and [`Runtimes_usage.md`](Runtimes_usage.md) landed. The agent-lens migration guide is **not** written — it needs the R3 gate met first, since it would be telling another project to depend on an unproven path |

**R1–R2 involve no spend at all** and are worth landing early: they are pure
computation and unlock `estimate` as a useful standalone tool.

### 6.1 What R1 settled

The subsystem exists but **cannot spend anything yet** — there is no real
backend, only `FakeRuntime`. That is deliberate: R1's job was to get the safety
model and the provider seam right while mistakes are still free.

**The safety rules are now enforced rather than described.** Four of the five
are implemented in `RuntimeManager` and each has tests:

- *No implicit spend* — the subsystem is off by default, `LLMCore.create()`
  builds the manager without contacting any backend, and `up()` raises
  `SpendNotConfirmedError` unless confirmation is explicit. `estimate()` is free
  and deliberately works **while disabled**, because deciding whether to spend
  should not require enabling spend.
- *No implicit persistence of spend* — state is written **before** provisioning
  returns, since the dangerous window is a crash between assignment and
  bookkeeping, where money burns and nothing knows. One indented-JSON file per
  runtime, so someone who suspects they are being billed can find out with `ls`
  and `cat`.
- *Bounded by default* — idle and hard deadlines come from config defaults, not
  from the caller remembering to pass them.
- *Fail closed* — if attach fails after `up()` succeeded, the runtime is
  released rather than left burning, and the error says so.

Rule 5 (no implicit secrets) stays with the backends, which own credentials.

**Three decisions worth recording:**

1. **`close()` detaches; it does not tear down.** A process exiting is not a
   reason to destroy compute someone is paying for and may still want, so
   `LLMCore.close()` unregisters the provider instances and leaves the state
   files. `down_all()` is the explicit way to stop spending. Getting this
   backwards would make every crashed script silently destroy a warm runtime —
   or, worse, make every clean exit look like it had.
2. **`DEGRADED` is a billing phase.** A broken runtime is still an assigned one,
   so `RuntimePhase.is_billing` includes it. The reaper and teardown both key
   off that property rather than off "is it working".
3. **A compute ceiling is checked before either deadline.** An idle reaper does
   not protect against a runtime that is *busy* in a loop, which is the
   expensive failure mode the reference design misses.

**Two robustness choices** came from asking what happens when the state
directory is already wrong: an unknown `phase` string parses as `DEGRADED`
rather than raising (a file written by a newer llmcore still describes a VM
burning money), and one corrupt record is skipped with a warning rather than
failing the whole listing (one bad file must not hide the runtimes still
running).

**The gate — dynamic provider registration — is met.** `attach()` registers the
runtime's endpoint as a `vllm`-type instance, marked `ephemeral=True` so the
provider manager tears it down with the rest, and `replace=True` so a legitimate
re-attach after a reconnect does not fail. Verified against the real
`ProviderManager`: `llm.runtimes.up(...)` yields a resolvable `VLLMProvider`
pointed at the tunnel URL, and `down()` unregisters it. `api_style` picks the
provider type, so a future recipe speaking TGI or llama.cpp attaches a different
type without touching this layer.

**Note on tooling:** the spec previously suggested `uv tool install
google-colab-cli`. llmcore's convention is pip, so the install hint is now
`pip install google-colab-cli`. The CLI (0.7.4) is installed in the shared venv
and ready for R3.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| **Runaway spend** — the defining risk | §2: explicit-action-only, idle reaper on, hard lifetime cap, orphan detection, state always inspectable |
| Colab CLI is **not installed** on this machine (`colab` not on PATH) and is Linux/macOS only | Discover at call time, fail with a `pip install google-colab-cli` hint; never a hard dependency of llmcore |
| Colab auth/quota errors (400/412) | Surface verbatim in `status` with the next action; prune the SKU ladder interactively |
| Upstream CLI is young; flags may move | Pin a tested CLI version range in the docs; parse `--json` output where offered, never scrape human text |
| Platform limits (~12 h, ~90 min idle) | Surface as ETAs; never circumvent |
| Tunnel dies silently | Liveness probe + `DEGRADED` state + cheap server-only restart |
| Drive cache corruption | Sentinel files per artifact; `cache gc`; re-download on sentinel mismatch |

---

## 8. Open questions

1. **Does the runtimes subsystem belong in llmcore core, or in an extra?**
   *Recommendation: `llmcore[runtimes]`* — it needs no new hard dependency, but
   the Colab backend shells out to an external CLI, which is unusual for llmcore
   and should be opt-in.
2. **Should `agent-lens` migrate in the same cycle,** or should llmcore ship R1–R4
   and let agent-lens migrate when convenient? The duplicated logic will drift.
3. **Where do the sizing heuristics live?** agent-lens has an opinionated,
   working sizer with an optional LLM plan review. Move the arithmetic to
   llmcore and leave the "lens" adversarial review in agent-lens?
4. **Multi-runtime routing** is a non-goal for v1, but should `RuntimeHandle`
   carry enough (cost/sku/latency) for a future router? *Recommendation: yes —
   the fields are free now and retrofitting them is not.*
5. **Colab auth** — needs a Google account OAuth via the CLI, not an API key.
   Confirm the intended account before R3, since it is the account that gets
   billed in compute units.

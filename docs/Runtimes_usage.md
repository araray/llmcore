# Remote GPU runtimes — usage guide

Size an open-weights model, provision compute, serve it, and reach it through
the same `llm.chat()` as any hosted API.

Three backends ship: **Colab**, **gpu.ai** and **DeepInfra**.

Design and safety rationale: the Colab runtime design spec.
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

## Pick a backend

| | Colab | gpu.ai | DeepInfra |
|---|---|---|---|
| Billed in | compute units | **US dollars**/hour | **US dollars**/hour |
| Price source | a table in llmcore | live, `GET /v1/pricing` | live, `GET /deploy/llm/gpu_availability` |
| Regions | none | 7, priced separately | none |
| Multi-GPU | no | yes | yes |
| Expires by itself | yes, on its own | only if you ask (`auto_terminate_hours`) | **no** |
| Logs through llmcore | yes | no (web console / SSH) | yes |
| Needs a key to size | no (CLI auth) | **no** — the catalogue is public | yes |

The thing to carry away from that table is the "expires by itself" row. A
forgotten Colab session stops on its own; a forgotten rental does not. Both
rental backends therefore set a platform-side lifetime where the platform
supports one, and `llmcore-runtimes status` reports instances llmcore did not
start as **orphans**, with the command to adopt and kill them.

llmcore never converts between compute units and dollars. There is no published
rate to convert with, so each backend quotes and bounds spend in the unit it is
actually billed in.

---

## Install

```bash
pip install "llmcore[runtimes]"      # or: pip install google-colab-cli
colab --help                         # the official Colab CLI must authenticate once
```

```toml
[runtimes]
enabled = true
default_backend = "colab"   # or "gpuai", or "deepinfra"

  [runtimes.defaults]
  confirm_spend = true
  idle_minutes = 45
  max_lifetime_minutes = 240

  [runtimes.gpuai]
  # GPUAI_API_KEY by default. Sizing needs no key at all.
  api_key_env_var = "GPUAI_API_KEY"
  # Optional: restrict which shapes the sizer will consider, cheapest first.
  # Omit it and the whole live catalogue is walked cheapest-first.
  # sku_ladder = ["1xa40", "1xa100_80gb", "2xa100_80gb"]

  [runtimes.deepinfra]
  api_key_env_var = "DEEPINFRA_API_KEY"
  # sku_ladder = ["1xA100-80GB", "2xA100-80GB", "1xH100-80GB"]
```

### Renting from gpu.ai

```bash
llmcore-runtimes estimate Qwen/Qwen2.5-7B-Instruct --context 32768 --backend gpuai
```

```
model      Qwen/Qwen2.5-7B-Instruct
shape      1xa40 in ca-central
fits       yes
burn rate  ~$0.50/hour while assigned

working:
  - 1xa40: ca-central at $0.49/hour; plus 100 GB disk at $0.014/hour; secure
  - pinned to offering 3a5ee74dcf97, so the launch cannot be placed on a
    pricier row than this one
```

Two details in that output are deliberate:

- **The rate includes the disk.** gpu.ai bills the instance filesystem
  separately from the GPU. Quoting only `price_per_hour` would understate the
  bill, which is the one direction a cost-bounding subsystem must not be wrong
  in.
- **The offering is pinned.** A catalogue row is a quote, not a booking.
  Without `offering_id`, the platform is free to place the instance on a
  pricier row than the one you approved.

gpu.ai serves the model with its own `vllm` template, which needs **at least 24
GB of VRAM** — smaller rungs are reported as unable to run it rather than
silently skipped.

### Renting from DeepInfra

DeepInfra sells compute twice, and only one of the two is a runtime:

- **Serverless inference**, billed per token — that is the `deepinfra`
  *provider*, not a runtime. Nothing to size, start or reap.
- **Dedicated LLM deployments**, billed per hour of reservation — that is the
  runtime this backend drives.
- **Container rentals** (`/v1/containers`) are raw GPU nodes with no server on
  them. llmcore does not drive these: it would have to install and expose a
  server itself, and at the time of writing DeepInfra offers them on B200 only,
  with no capacity available.

```bash
llmcore-runtimes estimate meta-llama/Llama-3.3-70B-Instruct --backend deepinfra
```

```
shape      4xA100-80GB
fits       yes
burn rate  ~$3.56/hour while assigned

working:
  - a dedicated deployment is billed for the reservation, not per token:
    stopping it is the only thing that stops the cost
```

Two DeepInfra-specific behaviours worth knowing:

- A deployment's name must be **owned** — `<your display name>/<model>` — and
  DeepInfra rejects anything else. llmcore reads the display name from
  `/v1/me`; override it with `runtimes.deepinfra.owner`.
- `down()` **deletes** the deployment, because DeepInfra reserves the hardware
  for as long as the deployment exists. Scaling it to zero is not the same as
  not paying for it. `down(release=False)` stops it instead and records the
  runtime as `detached`, meaning llmcore has let go but the cost may continue.

### Renting a bare gpu.ai machine

Everything above serves one model on a template. When you want the machine
instead — your own code, environments and server, reached over SSH — rent a
bare instance. It is a direct backend call, not a `RuntimeManager` verb: there
is no endpoint to attach as a provider, and you drive the box yourself.

```python
from llmcore.runtimes import Ownership, RuntimeStateStore
from llmcore.runtimes.gpuai import GpuAiRuntime, InstanceRequest

rt = GpuAiRuntime(
    state_store=RuntimeStateStore("/path/to/your/ledger"),   # not ~/.llmcore
    ownership=Ownership(name_prefix="myapp-"),
)

offers = await rt.offers(gpu_type="rtx_6000_ada")   # free, no key, cheapest first
for o in offers:
    print(o.offering_id, o.region, o.deployment_type, o.instant_boot,
          o.cost_per_hour(), o.environment and o.environment.cuda)

keys = await rt.ssh_keys()                         # or: await rt.create_ssh_key(name, pub)

request = InstanceRequest.from_offer(
    offers[0],
    environment="certified",          # or image="ghcr.io/you/img:tag"
    ssh_key_ids=(keys[0].id,),
    auto_terminate_hours=2,           # gpu.ai kills it even if you crash
    env={"MYAPP_ROOT": "/root/app"},  # values are never written to state
)
handle = await rt.rent(request, name="myapp-run-1", idempotency_key="run-1")
handle.metadata["ssh_command"]        # e.g. "ssh root@frp.gpu.ai -p 41022"
handle.metadata["hostname"], handle.metadata["port"]

await rt.terminate(handle.external_id)   # or: await rt.down("myapp-run-1")
```

- **What `offers()` keeps that the catalogue drops:** VM or container, boot
  class (`instant_boot`), capacity class, disk allowance and price, host
  specs, and the image's CUDA version and minimum driver. The cheapest row is
  often a VM that boots in 3–5 minutes; the next one up may boot in one.
- **`from_offer` pins the offering and uses its price as both ceilings**,
  rounded up to the cent. `rent()` refuses an invalid request (image *and*
  environment, port 22, reserved env names, a lifetime outside 1–720 h) before
  sending anything, and refuses a name that already records a live instance.
- **`rent()` waits for `running`**, which is gpu.ai's SSH-ready — not for an
  app URL, which a bare box never gets.
- **Pass your own `idempotency_key`** if you keep your own record of rentals:
  write it down before calling, and a retry after a crash returns the same
  instance rather than a second one.
- **The record is written before the create**, then updated with the operation
  id and the instance id. If the process dies mid-wait, `status()` and
  `down(name)` resolve the recorded operation and find the instance.
- **`terminate(id)` raises if the DELETE fails.** It is idempotent otherwise.

### Reading the account

```python
from llmcore.runtimes import openrouter_balance
from llmcore.runtimes.deepinfra import DeepInfraRuntime

balance = await rt.balance()                    # gpu.ai /billing/balance
balance.balance_usd, balance.burn_usd_per_hour, balance.runway_hours
await rt.spending_limit()                       # cap, spend, enforcement
report = await rt.usage(start="2026-10-01T00:00:00Z")
report.by_instance()                            # dollars per instance, from the meter

await DeepInfraRuntime().balance()              # /payment/checklist
await openrouter_balance()                      # OPENROUTER_API_KEY
```

**None of these raise.** A failed read returns the same type with `error` set
and every figure `None`, so a panel showing three vendors shows "could not
read" for one instead of failing. `None` always means unknown: gpu.ai reports
a `null` burn rate when it cannot tell, and that is never shown as zero.

### Shared accounts: whose instance is it?

`status()` reports any instance the state store does not know as an **orphan**
with an adopt hint. If the same account is also used by hand or by another
tool, give the backend an `Ownership` rule (`name_prefix` and/or `known_ids`,
or `runtimes.gpuai.owned_name_prefix` / `owned_ids`): strays it claims stay
orphans, and everything else is reported as **not ours** (`owned=False`), with
no adopt hint.

### Credentials are never in state files

A handle's credentials — DeepInfra's account key, a gpu.ai template's tunnel
password — live in `handle.secrets`, in memory only. A handle read back from a
state file has none; `await backend.credentials(handle)` re-derives them.
State files written by older versions are scrubbed the next time they are
saved.

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

## Accrued spend

`status` reports what a runtime has cost so far, in its own unit:

```
NAME             PHASE      SKU      MODEL                        EXPIRES  SPENT
llama            ready      4xA100-  araray/Llama-3.3-70B         3h 12m   $4.15
```

A `-` in that column means the backend does not report consumption. It does
**not** mean the runtime is free.

The two backends get the figure differently, and the difference matters:

- **DeepInfra** has a meter (`/payment/usage/rent`), so llmcore reads metered
  seconds. A deployment that was stopped and restarted has billed for less time
  than it has existed, and only the meter knows that.
- **gpu.ai** publishes no per-instance running total on the instance, so
  `status` derives `rate × elapsed` from its creation time — which
  over-estimates (provisioning is not billed, and unreachable windows are
  credited back), so a ceiling fires early rather than late. What was actually
  billed is on the account meter: `await rt.usage(group_by="instance_id")`.

`max_spend` is compared against this figure, in the backend's own unit:

```python
await llm.runtimes.up(
    "Qwen/Qwen2.5-7B-Instruct",
    name="qwen",
    backend="gpuai",
    confirm_spend=True,
    max_spend=5.00,        # dollars, on this backend
    max_lifetime_minutes=120,
)
```

It is the only guard against the expensive failure mode: a runtime that is
*busy* in a loop, which the idle reaper never touches.

> `max_compute_units` is the old name for `max_spend`, from when Colab was the
> only backend. It still works and warns.

---

## Before you spend: `doctor`

```bash
llmcore-runtimes doctor            # every backend
llmcore-runtimes doctor --backend gpuai
```

```
[ok  ] subsystem                  runtimes.enabled is true
[ok  ] state: directory           ~/.llmcore/runtimes is writable
[ok  ] billing                    nothing is running, as far as llmcore sees
[ok  ] spend: usd today           $17.50 plus 1 unpriced run(s) of $20.00
[FAIL] colab: cli                 'colab' was not found on PATH
                                  -> install it, or set the backend's cli_path
[ok  ] gpuai: catalogue           19 priced shapes, cheapest 1xa40 at $0.60/hour
[ok  ] gpuai: credential          GPUAI_API_KEY is set
[ok  ] gpuai: instances           no instances on the account
```

Free, read-only, and it works when the rest of llmcore does not. The exit code
is the worst finding — **2** if something blocks, **1** if something is billing
or degraded, **0** if clean — so `doctor && up` will not provision into a
broken state.

Severity is graded by *consequence*, not by state:

| | meaning |
|---|---|
| `FAIL` | provisioning cannot work |
| `warn` | it can, but something will be missing |
| `$$$` | something may be costing money **right now** — outranks both |

The first live run of this found a real problem: llmcore's default Colab
`cli_path` is a bare `colab`, so on a machine where that CLI lives in a venv
llmcore could not drive Colab at all — and would only have said so at
provision time, after the decision to spend.

A credential check never prints the value, only whether the variable is set.

---

## A daily budget

`max_spend` bounds **one runtime**. `[routing.budget]` bounds **one turn**.
Neither bounds a day — so ten runtimes, each politely under its own ceiling,
could spend ten times what you intended and nothing would object.

```toml
[runtimes.budget]
max_usd_per_day = 20.00              # gpu.ai, DeepInfra
max_compute_units_per_day = 150      # Colab
```

**Two dials, because one cap cannot span two units.** Colab bills compute units
with no published dollar value; the rental backends bill dollars. There is no
defensible rate between them, so there is no single number.

The cap is checked **before** `up()` contacts the backend — a cap applied after
compute is assigned is not a cap — and refuses with
`DailyBudgetExceededError`, its own type so you can catch "cannot afford this"
separately from "this is broken".

```bash
llmcore-runtimes spend              # per day, per unit, last 7 days
```

```
DAY          UNIT            RUNS              SPENT
2026-10-10   usd                3 $17.50 plus 1 unpriced run(s)
2026-10-09   compute-unit       1    42.00 compute units
```

The ledger is append-only JSON lines at `<state_dir>/spend.jsonl` — readable
with `cat`, and a crash mid-write costs you the last line rather than the file.
An entry is added when a runtime is **released**, which is the moment its cost
stops moving.

**Unknown spend is not zero.** A run whose cost could not be determined is
recorded as *unpriced*, never as `0.00`, and every total that contains one says
so. A day of unpriceable runtimes must not read as a free day — and when the
cap refuses, it tells you the real figure is higher than the one it quoted.

Start with no cap and read `spend` for a week. A number nobody chose is worse
than no number: the agent circuit breaker shipped with a `$1.00` default that
would have cut off half of this project's normal turns.

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

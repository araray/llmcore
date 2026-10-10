"""Every llmcore documentation diagram, as code.

One function per diagram. Each returns a :class:`~tools.diagrams.dsl.Diagram`
and registers it under the slug that becomes its filename, so adding a
diagram means adding a function and referencing ``docs/diagrams/<slug>.svg``
from the prose.
"""

from __future__ import annotations

from .dsl import ACCENT, INK, INK_SOFT, LINE, Diagram

DIAGRAMS: dict[str, Diagram] = {}


def reg(slug: str, d: Diagram) -> Diagram:
    """Register *d* under *slug*, which becomes ``docs/diagrams/<slug>.svg``."""
    DIAGRAMS[slug] = d
    return d


# ---------------------------------------------------------------- 02 / cycle
def cognitive_cycle():
    W, H = 880, 330
    d = Diagram(W, H, "The cognitive cycle",
                "Eight phases run in a loop: perceive, plan, think, validate, "
                "act, observe, reflect, update. The top row runs left to right, "
                "the bottom row right to left, and update feeds back into "
                "perceive.")
    d.bg()
    d.text(W / 2, 42, "Cognitive cycle", size=17, bold=True)
    d.text(W / 2, 62, "one iteration of an agent run", size=12, fill=INK_SOFT)

    top = [("PERCEIVE", "analyse goal"), ("PLAN", "form strategy"),
           ("THINK", "decide action"), ("VALIDATE", "check validity")]
    bot = [("ACT", "execute"), ("OBSERVE", "read results"),
           ("REFLECT", "learn"), ("UPDATE", "state & memory")]
    bw, bh, gap = 168, 72, 60
    x0 = (W - (4 * bw + 3 * (gap - bw + bw))) / 2
    x0 = (W - (4 * bw + 3 * 24)) / 2
    y_top, y_bot = 92, 204

    for i, (name, sub) in enumerate(top):
        x = x0 + i * (bw + 24)
        d.box(x, y_top, bw, bh, tone="agent")
        d.text(x + bw / 2, y_top + 30, name, size=13.5, bold=True)
        d.text(x + bw / 2, y_top + 51, sub, size=11.5, fill=INK_SOFT)
        if i < 3:
            d.arrow(x + bw + 3, y_top + bh / 2, x + bw + 21, y_top + bh / 2,
                    stroke=ACCENT["agent"][0])
    for i, (name, sub) in enumerate(bot):
        x = x0 + (3 - i) * (bw + 24)
        d.box(x, y_bot, bw, bh, tone="agent")
        d.text(x + bw / 2, y_bot + 30, name, size=13.5, bold=True)
        d.text(x + bw / 2, y_bot + 51, sub, size=11.5, fill=INK_SOFT)
        if i < 3:
            d.arrow(x - 3, y_bot + bh / 2, x - 21, y_bot + bh / 2,
                    stroke=ACCENT["agent"][0])

    right = x0 + 3 * (bw + 24) + bw / 2
    left = x0 + bw / 2
    d.arrow(right, y_top + bh + 3, right, y_bot - 3, stroke=ACCENT["agent"][0])
    d.arrow(left, y_bot - 3, left, y_top + bh + 3, stroke=ACCENT["agent"][0])
    # Beside the return arrow rather than centred on it, where it was
    # crossed by the line.
    d.text(left - 12, y_bot - 14, "next iteration", size=11, fill=INK_SOFT,
           anchor="end")
    return reg("cognitive-cycle", d)


# ------------------------------------------------- 07 / cycle, with the loop
def cognitive_cycle_detail():
    W, H = 760, 532
    d = Diagram(W, H, "The cognitive cycle in detail",
                "Plan runs once at the start. Think, act, observe and reflect "
                "then loop until the agent finishes or hits its iteration "
                "limit, at which point it returns a result.")
    d.bg()
    d.text(W / 2, 42, "Cognitive cycle", size=17, bold=True)
    d.text(W / 2, 62, "plan once, then loop until finished", size=12, fill=INK_SOFT)

    steps = [("PLAN", "create a high-level strategy, once at the start"),
             ("THINK", "decide the next action from current state"),
             ("ACT", "execute the chosen tool"),
             ("OBSERVE", "process the tool result"),
             ("REFLECT", "update understanding, log the episode")]
    bx, bw, bh = 70, 132, 46
    y0, step = 96, 66
    for i, (name, note) in enumerate(steps):
        y = y0 + i * step
        tone = "api" if i == 0 else "agent"
        d.box(bx, y, bw, bh, tone=tone)
        d.text(bx + bw / 2, y + bh / 2 + 4.5, name, size=13, bold=True)
        d.text(bx + bw + 22, y + bh / 2 + 4, note, size=12, fill=INK_SOFT,
               anchor="start")
        if i:
            d.arrow(bx + bw / 2, y - step + bh + 3, bx + bw / 2, y - 3)

    # The loop back from reflect to think.
    last_y = y0 + 4 * step
    think_y = y0 + step
    lane = bx - 30
    d.line(bx, last_y + bh / 2, lane, last_y + bh / 2, stroke=ACCENT["agent"][0])
    d.line(lane, last_y + bh / 2, lane, think_y + bh / 2, stroke=ACCENT["agent"][0])
    d.arrow(lane, think_y + bh / 2, bx - 3, think_y + bh / 2, stroke=ACCENT["agent"][0])
    d.vtext(lane - 11, (think_y + last_y) / 2 + bh / 2, "not finished", size=11,
            fill=ACCENT["agent"][0])

    # Exit.
    exit_y = last_y + bh + 44
    d.arrow(bx + bw / 2, last_y + bh + 3, bx + bw / 2, exit_y - 3)
    d.box(bx, exit_y, bw, bh, tone="data")
    d.text(bx + bw / 2, exit_y + bh / 2 + 4.5, "RESULT", size=13, bold=True)
    d.text(bx + bw + 22, exit_y + bh / 2 + 4,
           "finished, or the iteration ceiling was reached", size=12,
           fill=INK_SOFT, anchor="start")
    return reg("cognitive-cycle-detail", d)


# --------------------------------------------------------- 06 / agent flow
def agent_data_flow():
    W, H = 760, 684
    d = Diagram(W, H, "Agent run data flow",
                "A user goal enters AgentManager.run_agent_loop(), which runs "
                "plan, think, act, observe and reflect in sequence, looping "
                "until it finishes or reaches its iteration limit.")
    d.bg()
    d.text(W / 2, 40, "Agent run", size=17, bold=True)
    d.text(W / 2, 59, "from a goal to a result", size=12, fill=INK_SOFT)

    cx, bw, bh = 220, 210, 52
    bx = cx - bw / 2
    d.text(cx, 92, "User goal", size=13, bold=True)
    d.arrow(cx, 100, cx, 118)

    d.box(bx, 124, bw, bh + 10, tone="api")
    d.text(cx, 150, "AgentManager", size=13, bold=True)
    d.text(cx, 170, ".run_agent_loop()", size=11.5, mono=True, fill=INK_SOFT)

    steps = [("Plan", "create a high-level strategy"),
             ("Think", "decide the next action or tool"),
             ("Act", "execute the tool, in a sandbox"),
             ("Observe", "process the tool result"),
             ("Reflect", "update state, log the episode")]
    y0, step = 214, 76
    for i, (name, note) in enumerate(steps):
        y = y0 + i * step
        d.box(bx, y, bw, bh, tone="agent")
        d.text(cx, y + bh / 2 + 4.5, name, size=13, bold=True)
        d.arrow(bx + bw + 4, y + bh / 2, bx + bw + 40, y + bh / 2, stroke=LINE)
        d.text(bx + bw + 50, y + bh / 2 + 4, note, size=12, fill=INK_SOFT,
               anchor="start")
        if i:
            d.arrow(cx, y - step + bh + 3, cx, y - 3)
    d.arrow(cx, 124 + bh + 10 + 3, cx, y0 - 3)

    last = y0 + 4 * step + bh
    d.text(cx, last + 26, "loop until finished or the iteration ceiling is hit",
           size=11.5, fill=ACCENT["agent"][0])
    d.arrow(cx, last + 34, cx, last + 56)
    d.text(cx, last + 76, "Final result", size=13, bold=True, fill=ACCENT["data"][0])
    return reg("agent-data-flow", d)


# ------------------------------------------------------- 08 / sandbox walls
def sandbox_isolation():
    W, H = 780, 420
    d = Diagram(W, H, "Sandbox isolation",
                "Code an agent decides to run is routed into a Docker or VM "
                "sandbox with an isolated filesystem, capped resources and "
                "optional network isolation, so it fails there rather than on "
                "the host.")
    d.bg()
    d.labelled_box(24, 22, W - 48, H - 44, "", tone="neutral", dashed=True)
    d.text(44, 48, "HOST SYSTEM", size=12.5, bold=True, fill=INK_SOFT, anchor="start")

    d.box(52, 66, W - 104, 76, tone="api")
    d.text(W / 2, 92, "Your application", size=13, bold=True)
    d.text(W / 2, 118, "agent.execute_python(\"import os; os.remove('/')\")",
           size=11.5, mono=True, fill=INK_SOFT)

    d.arrow(W / 2, 146, W / 2, 178, stroke=ACCENT["risk"][0])
    d.text(W / 2 + 14, 168, "routed to the sandbox", size=11.5,
           fill=ACCENT["risk"][0], anchor="start")

    d.box(52, 184, W - 104, 162, tone="risk")
    d.text(W / 2, 210, "SANDBOX  ·  Docker or VM", size=13, bold=True)
    inner_y = 224
    d.box(76, inner_y, W - 152, 104, tone="risk", fill="#ffffff", dashed=True)
    facts = ["isolated filesystem — no access to host files",
             "capped CPU and memory",
             "optional network isolation"]
    for i, fact in enumerate(facts):
        d.text(100, inner_y + 26 + i * 21, "•", size=12, fill=ACCENT["risk"][0],
               anchor="start")
        d.text(116, inner_y + 26 + i * 21, fact, size=12, fill=INK_SOFT,
               anchor="start")
    d.text(W / 2, inner_y + 94, "code executes here, and fails here", size=12,
           bold=True, fill=ACCENT["risk"][0])
    d.text(W / 2, 372, "the host is untouched", size=12.5, bold=True,
           fill=ACCENT["data"][0])
    return reg("sandbox-isolation", d)


# ----------------------------------------------------- 03 / sandbox registry
def sandbox_registry():
    W, H = 780, 380
    d = Diagram(W, H, "Sandbox registry",
                "The sandbox registry manages sandbox lifecycles across two "
                "providers, Docker and VM, with an output tracker for file "
                "lineage and an ephemeral manager for cleanup and timeouts.")
    d.bg()
    d.band(24, 22, W - 48, 56)
    d.text(W / 2, 46, "Sandbox registry", size=17, bold=True)
    d.text(W / 2, 66, "manages sandbox lifecycle", size=12, fill=INK_SOFT)

    cells = [
        ("Docker provider", ["container isolation", "image management",
                             "resource limits", "output capture"], "risk"),
        ("VM provider", ["SSH-based access", "full VM isolation",
                         "network separation", "persistent storage"], "risk"),
        ("Output tracker", ["file lineage", "execution logs",
                            "artifact collection"], "data"),
        ("Ephemeral manager", ["resource cleanup", "timeout handling",
                               "state management"], "neutral"),
    ]
    bw, bh = 336, 118
    for i, (title, items, tone) in enumerate(cells):
        x = 36 + (i % 2) * (bw + 32)
        y = 98 + (i // 2) * (bh + 22)
        d.box(x, y, bw, bh, tone=tone)
        d.text(x + 18, y + 26, title, size=13, bold=True, anchor="start")
        for j, item in enumerate(items):
            d.text(x + 22, y + 50 + j * 19, "•", size=11, fill=LINE, anchor="start")
            d.text(x + 36, y + 50 + j * 19, item, size=11.5, fill=INK_SOFT,
                   anchor="start")
    return reg("sandbox-registry", d)


# ---------------------------------------------------- 05 / component layers
def component_hierarchy():
    W, H = 760, 420
    d = Diagram(W, H, "Component hierarchy",
                "Four layers: your application, LLMCore with its chat, storage "
                "and memory managers, the agent system built on them, and the "
                "sandbox layer beneath.")
    d.bg()
    layers = [
        ("Your application", [], "neutral", 48),
        ("LLMCore", ["LLMCore (chat)", "Storage manager", "Memory manager"], "api", 96),
        ("Agent system", ["Cognitive cycle", "Tool manager", "Sandbox integration"],
         "agent", 96),
        ("Sandbox layer", ["Sandbox registry", "Docker provider", "VM provider"],
         "risk", 96),
    ]
    y = 26
    for title, cells, tone, h in layers:
        d.box(28, y, W - 56, h, tone=tone)
        d.text(W / 2, y + 26, title, size=14, bold=True)
        if cells:
            cw = (W - 56 - 4 * 18) / 3
            for i, cell in enumerate(cells):
                cx = 28 + 18 + i * (cw + 18)
                d.box(cx, y + 40, cw, 40, tone="neutral", fill="#ffffff")
                d.text(cx + cw / 2, y + 64, cell, size=11.5, fill=INK)
        y += h + 14
    return reg("component-hierarchy", d)


# -------------------------------------------------------- 01 / architecture
def architecture():
    W, H = 980, 700
    d = Diagram(W, H, "llmcore architecture",
                "The LLMCore facade over four core managers and their backing "
                "subsystems, two optional subsystems that stay off until "
                "configured, the agent system, the sandbox system and the "
                "supporting registries.")
    d.bg()
    d.band(24, 22, W - 48, 58)
    d.text(W / 2, 48, "LLMCore API facade", size=17, bold=True)
    d.text(W / 2, 68, "llmcore.api.LLMCore", size=12, mono=True, fill=INK_SOFT)

    # Core managers and what each sits on.
    core = [("Provider manager", ["23 vendors", "direct REST", "+ SDK fallbacks"]),
            ("Session manager", ["SQLite", "Postgres", "JSON"]),
            ("Memory manager", ["ChromaDB", "pgvector", "external bridge"]),
            ("Embedding manager", ["SentenceTransformers", "OpenAI", "Google"])]
    cw = (W - 48 - 3 * 16) / 4
    y_top, y_bot = 96, 172
    for i, (title, items) in enumerate(core):
        x = 24 + i * (cw + 16)
        d.box(x, y_top, cw, 46, tone="api")
        d.text(x + cw / 2, y_top + 28, title, size=12.5, bold=True)
        d.arrow(x + cw / 2, y_top + 46 + 2, x + cw / 2, y_bot - 3, stroke=LINE)
        d.box(x, y_bot, cw, 78, tone="neutral", fill="#ffffff")
        for j, item in enumerate(items):
            d.text(x + cw / 2, y_bot + 24 + j * 19, item, size=11, mono=True,
                   fill=INK_SOFT)

    # Optional subsystems.
    oy = 278
    d.text(24, oy - 8, "optional subsystems — off until configured", size=11.5,
           fill=INK_SOFT, anchor="start")
    opt = [("Media manager", ["image / audio / video",
                              "jobs + webhooks  ·  artifacts"]),
           ("Runtimes manager", ["size → up → attach as a provider",
                                 "idle reaper  ·  spend ceiling"])]
    ow = (W - 48 - 16) / 2
    for i, (title, items) in enumerate(opt):
        x = 24 + i * (ow + 16)
        d.box(x, oy, ow, 74, tone="data", dashed=True)
        d.text(x + ow / 2, oy + 24, title, size=12.5, bold=True)
        for j, item in enumerate(items):
            d.text(x + ow / 2, oy + 45 + j * 17, item, size=11, mono=True,
                   fill=INK_SOFT)

    # Agent system.
    ay = 370
    d.box(24, ay, W - 48, 136, tone="agent")
    d.text(W / 2, ay + 26, "Agent system", size=14, bold=True)
    d.box(44, ay + 38, W - 88, 36, tone="agent", fill="#ffffff")
    d.text(W / 2, ay + 61,
           "PERCEIVE → PLAN → THINK → VALIDATE → ACT → OBSERVE → REFLECT → UPDATE",
           size=11.5, mono=True, fill=INK)
    chips = ["Goal classifier", "Fast-path executor", "Circuit breaker",
             "HITL manager", "Persona manager", "Prompt library",
             "Activity system", "Capability checker"]
    chw = (W - 88 - 3 * 12) / 4
    for i, chip in enumerate(chips):
        x = 44 + (i % 4) * (chw + 12)
        y = ay + 84 + (i // 4) * 26
        d.box(x, y, chw, 22, tone="neutral", fill="#ffffff", radius=5)
        d.text(x + chw / 2, y + 15, chip, size=10.5, fill=INK_SOFT)

    # Sandbox + supporting.
    sy = 522
    d.box(24, sy, W - 48, 58, tone="risk")
    d.text(W / 2, sy + 24, "Sandbox system", size=13, bold=True)
    d.text(W / 2, sy + 44, "Docker provider  ·  VM provider  ·  registry  ·  output tracker",
           size=11.5, mono=True, fill=INK_SOFT)

    py = 596
    d.text(W / 2, py + 16, "Supporting systems", size=13, bold=True)
    sup = ["Model card registry", "Observability", "Tracing", "Logging config"]
    sw = (W - 48 - 3 * 14) / 4
    for i, name in enumerate(sup):
        x = 24 + i * (sw + 14)
        d.box(x, py + 28, sw, 38, tone="neutral", fill="#ffffff")
        d.text(x + sw / 2, py + 52, name, size=11.5, fill=INK_SOFT)
    return reg("architecture", d)

# ------------------------------------------------- model_cards.md / registry
def model_card_registry():
    """docs/model_cards.md -- the registry and its two card tiers."""

    W, H = 780, 392
    d = Diagram(W, H, "ModelCardRegistry architecture",
                "The ModelCardRegistry singleton resolves a model card from two "
                "tiers: built-in cards packaged with llmcore, and user cards under "
                "~/.config/llmcore that override them. Three indexes map provider "
                "and model id, aliases, and lowercased ids onto cards.")
    d.bg()

    # Header
    d.band(24, 22, W - 48, 58)
    d.text(W / 2, 48, "ModelCardRegistry", size=17, bold=True)
    d.text(W / 2, 68, "process-wide singleton", size=12, fill=INK_SOFT)
    d.line(24, 92, W - 24, 92, stroke="#e2e8f0", width=1)

    # The two tiers
    tier_y, tier_h, tier_w = 110, 104, 250
    # Three stacked labels per tier, so the box draws without `labelled_box`'s
    # single centred title -- which collided with the lines below it.
    for x0, title, path, prio, tone, prio_fill in (
        (56, "Built-in cards", "llmcore/model_cards/", "lower priority",
         "neutral", INK_SOFT),
        (W - 56 - tier_w, "User cards", "~/.config/llmcore/", "higher priority",
         "data", ACCENT["data"][0]),
    ):
        d.box(x0, tier_y, tier_w, tier_h, tone=tone)
        cx = x0 + tier_w / 2
        d.text(cx, tier_y + 32, title, size=13.5, bold=True)
        d.text(cx, tier_y + 58, path, size=11.5, mono=True, fill=INK_SOFT)
        d.text(cx, tier_y + 84, prio, size=11.5, fill=prio_fill)

    # Override arrow, right to left: the user tier wins.
    mid_y = tier_y + tier_h / 2
    d.arrow(W - 56 - tier_w - 10, mid_y, 56 + tier_w + 12, mid_y, stroke=ACCENT["data"][0])
    d.text(W / 2, mid_y - 14, "overrides", size=12, bold=True, fill=ACCENT["data"][0])
    # The qualifier sits under both boxes rather than in the 168px gap between
    # them, where it collided with the right-hand border.
    d.text(W / 2, tier_y + tier_h + 22, "when both define the same (provider, model_id)",
           size=11.5, mono=True, fill=INK_SOFT)

    # Indexes
    ix_y = 254
    d.line(24, ix_y, W - 24, ix_y, stroke="#e2e8f0", width=1)
    d.text(56, ix_y + 26, "Indexes built on load", size=13, bold=True, anchor="start")
    rows = [
        "provider  →  model_id  →  ModelCard",
        "alias  →  (provider, model_id)",
        "lowercased model_id  →  [(provider, model_id), …]",
    ]
    for i, row in enumerate(rows):
        y = ix_y + 52 + i * 26
        d.text(62, y, "•", size=13, fill=LINE, anchor="start")
        d.text(80, y, row, size=12, mono=True, fill=INK_SOFT, anchor="start")
    return reg("model-card-registry", d)


def build_all() -> dict[str, Diagram]:
    """Construct every diagram, in declaration order."""
    DIAGRAMS.clear()
    for make in (
        architecture,
        cognitive_cycle,
        cognitive_cycle_detail,
        agent_data_flow,
        sandbox_isolation,
        sandbox_registry,
        component_hierarchy,
        model_card_registry,
    ):
        make()
    return DIAGRAMS

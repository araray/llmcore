# Examples

## `lane_eval_starter.jsonl` — a starting point for measuring classifiers

llmcore makes **no accuracy claim** for any classifier, because none has been
validated on anyone's real traffic and any number would be invented. This file
is the smallest useful step toward replacing that gap with a measurement.

It is 29 hand-written cases across five lanes (`trivial`, `standard`, `deep`,
`code`, `private`). It is **not** a benchmark, and it should not be treated as
one:

- the labels are one person's judgement about what *should* be cheap;
- the prompts are invented, not sampled from traffic;
- 29 cases is enough to catch an obviously wrong chain and nothing more.

What it is good for is showing the shape, so replacing it with your own is a
small job:

```jsonl
{"prompt": "rename this variable", "expected": "trivial"}
{"prompt": "prove this lemma", "expected": "deep", "note": "maths"}
```

Then:

```bash
llmcore-routing eval my_traffic.jsonl \
  --lane-order trivial,code,standard,deep \
  --chain
```

```
29 cases (29 labelled), lanes cheapest-first: trivial < code < standard < deep

heuristic
  answered   21/29 (72% coverage, 8 abstentions)
  agreement  14/21 (67%)
  too cheap  5  <- these produce bad answers
  too dear   2  <- these only cost money
  latency    p50 0 ms, p95 1 ms
```

**Read the two misroute lines, not the accuracy.** A classifier that is 67%
accurate but never routes too cheap is usable; the same 67% erring downward may
not be. `--show-misroutes N` prints the individual prompts that went too cheap,
which is where the useful information is.

A few prompts in the starter set exist specifically to catch the trap that
caught llmcore's own heuristic: **prompt length does not predict request
complexity.** "Write a 2000-word essay comparing two schools of jurisprudence"
is ten tokens and is not a cheap request.

## About the `private` rows

The free `heuristic` classifier scores **0 out of 4** on them, and that is
expected rather than a defect: it is documented as not looking for personal
data, because patterns that find names and addresses also flag most ordinary
English. On the first run of this harness it routed

> "Here is my patient record: John Doe, DOB 1971-03-02, diagnosed with
> hypertension. Summarise it."

to the **trivial** lane, because "summarise" is a simple-task verb.

That is only acceptable because **the privacy guarantee does not depend on the
classifier.** Transforms run *after* target selection and can change the
destination, so a PII prompt misrouted to the cheap lane is still constrained
to a local-only pool before anything is sent. There is a test asserting exactly
this — including that the classifier really does get it wrong, so the premise
cannot rot silently (`tests/routing/test_manager.py`,
`TestLayeringSurvivesAMisclassification`).

So: keep the `private` rows if you want to know whether a *classifier* can
spot sensitive prompts (a local encoder can; the heuristic cannot). Drop them
if you only care about cost tiers. Either way, do not read a low score on them
as the privacy path being broken.

## Why the chain answers so rarely

In the run above the chain has 17% coverage, far below the heuristic alone.
That is the confidence floor working: the heuristic's "nothing stood out"
answer is deliberately low-confidence (0.4), so in a chain it falls below
`min_confidence` and abstains, and the request falls through to the default
pool. A chain is meant to be mostly abstentions with a cheap classifier at the
front — the alternative is routing confidently on a weak signal.

An unlabelled `.txt` file (one prompt per line) is also accepted. That measures
coverage and latency on real traffic without anyone having to label it first,
which is a reasonable thing to do before deciding whether labelling is worth it.

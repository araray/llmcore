# Scheduled jobs that keep llmcore's data current

These are kairos workflow definitions, versioned here because they describe
how *this* repo's data gets refreshed. The code they invoke deliberately
lives outside llmcore, in `/av/repos/ai-tools`: those jobs reach out to the
network on a timer, and that risk does not belong inside a library.

So each file here is the authoritative copy of a definition that is also
checked into the tool's own repo. Keep them in step; the copy that actually
runs is the one installed in kairos.

## pricewatch.yaml

Refreshes model pricing daily from vendor APIs and vendor pricing pages.

```bash
cp tools/kairos/pricewatch.yaml /av/conf/kairos/workflows/
kairos workflows show pricewatch
kairos workflows run pricewatch      # run it now rather than waiting
```

It writes complete user-override cards into
`~/.config/llmcore/model_cards/`, which the registry already loads on top
of the packaged cards, so llmcore needs no change to pick them up.

**Why this exists.** The packaged cards are mostly unpriced, and inverted:
`generated` cards refresh from provider `/v1/models` endpoints, which do
not return prices, while hand-written `builtin` cards carry pricing but
never refresh. Measured 2026-10-01: 1484 of 2319 cards unpriced (64%),
including every model in real observed traffic. That also means the routing
subsystem's `lowest_cost` strategy — which correctly ranks unpriced targets
last — was ranking the most-used models last.

**Two things to know before editing it.**

`jobs` is a **mapping** keyed by job name. The list form shown in kairos's
own USAGE.md is rejected by the loader.

Do not add `condition: 'job("fetch").last_success'` to the `apply` job.
That KEL predicate asks about the *previous completed run* of fetch, not
the current one, so a single held run makes `apply` skip indefinitely even
after fetch recovers. The `needs:` edge is the correct intra-run gate —
kairos already skips a job whose dependency failed in this run.

See `/av/repos/ai-tools/README.md` for the design: route-aware price
attribution, the refusal rules, and the diff gate.

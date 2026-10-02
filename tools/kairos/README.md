# Scheduled jobs that keep llmcore's data current

kairos workflow definitions, versioned here because they describe how this
repo's packaged data gets refreshed. The code they invoke lives in a separate
tool: these jobs reach out to the network on a timer, and that dependency
does not belong inside the library.

The copies here are machine-neutral — they call the tool from `PATH` and let
it locate the card tree. A deployment will usually want absolute paths and a
logging wrapper; keep those in the installed copy.

## pricewatch.yaml

Refreshes model pricing daily from vendor APIs and vendor pricing pages, and
writes complete user-override cards into `~/.config/llmcore/model_cards/`,
which `ModelCardRegistry` already loads on top of the packaged cards. No
llmcore code change is needed for them to take effect.

```bash
cp tools/kairos/pricewatch.yaml <your-kairos-workflows-dir>/
kairos workflows show pricewatch
kairos workflows run pricewatch      # run now instead of waiting for the trigger
```

Override cards are written whole, not as a pricing fragment: user cards
*replace* packaged cards by `model_id` rather than merging into them, so a
pricing-only card would drop the model's context window and capabilities.

### Two things to know before editing it

`jobs` is a **mapping** keyed by job name. The list form in kairos's own
USAGE.md is rejected by the loader.

Do not add `condition: 'job("fetch").last_success'` to the `apply` job. That
expression asks about the *previous completed run* of fetch, not the current
one, so a single held run makes `apply` skip indefinitely even after fetch
recovers. The `needs:` edge is the correct intra-run gate.

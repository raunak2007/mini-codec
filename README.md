# skydiscover-evaluator-integrity

Does an LLM-driven program-search loop notice when its evaluator is lying to it? A dead-parameter injection experiment
on [SkyDiscover](https://github.com/skyrl-org/skydiscover), UC Berkeley's evolutionary search framework for
LLM-generated programs.

## The experiment

SkyDiscover scores candidate programs with a containerized evaluator and evolves the population toward higher scores.
The framework has a train mode (the score the search loop sees) and a test mode (a held-out check on the same
programs). That split makes a clean experiment possible: perturb the evaluator's score **in train mode only**, and use
the framework's own test mode, untouched, as ground truth.

- **Injection**: the train-mode score is inflated by a term that depends on a parameter the program's real behaviour
  never uses (a "dead" parameter). Gaming it raises the reported score and does nothing for true quality.
- **Control**: identical search with no injection.
- **Measurement**: the gap between reported (train) and true (test) quality of the best program, tracked over
  iterations.
- **Setting**: search depth 15, three independent runs per condition.

## Results

| condition | reported minus true quality (best program, depth 15) |
|---|---|
| control | 0.000, 0.000, 0.000 |
| injected | +0.089 (mean of three runs) |

In the control runs the gap is exactly zero: reported and true quality agree throughout. In the injected runs the
search finds the dead parameter and exploits it. True quality peaks at iteration 8 and then flattens or declines while
the reported score keeps climbing to the end of the search; the search loop has no mechanism that would notice.

`runlogs/divergence.png` shows the two curves per condition.

## Incidental findings

Two upstream defects surfaced while setting this up, both reproducible from the shipped code:

1. Malformed SEARCH/REPLACE diffs from the model are applied anyway, writing conflict markers into candidate programs,
   which then fail evaluation for a reason unrelated to the search.
2. The shipped `text_similarity` example uses an unseeded `random.sample`, so its scores are not reproducible run to
   run.

## Why it matters

Any evaluator that a search or RL loop optimises against becomes a target. This experiment is the small, fully
controlled version of that problem: a known, injected flaw in the reward, a held-out check the loop cannot see, and a
measurement of how far the loop's self-report drifts from the truth. The same design applies to agent benchmarks where
the evaluation is itself a program the agent can probe.

## Layout

- `experiments/`: injection and control configurations, and the evaluator wrapper that applies the perturbation in
  train mode only.
- `runlogs/`: per-run logs of reported and true quality, and `divergence.png`.
- `repro/`: scripts to rerun both conditions against an upstream SkyDiscover checkout.

The upstream framework is not vendored here; `repro/` expects a separate clone.

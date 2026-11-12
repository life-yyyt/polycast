# Design notes

Short record of the decisions that were not obvious, and what they cost.

## 1. The panel runs in parallel and cannot see each other

Panel members run concurrently against the same prompt and never receive each
other's output. This is the single most consequential choice in the codebase.

The alternative — a debate loop where each analyst sees the previous answers —
produces tighter agreement and usually higher confidence. It also produces
worse forecasts, because the panel converges on the first confident-sounding
argument rather than on the evidence. The `agreement` we report is then an
artefact of the dialogue, not a measure of genuine uncertainty.

Cost: we lose the ability to make analysts respond to each other's strongest
point. The red team exists partly to recover that, with a bounded shift.

## 2. The red team runs after the panel, not beside it

The critic receives the panel mean, every analyst's probability and reasoning,
and the resolution criteria. It returns a bounded shift in `[-0.3, 0.3]`.

Running it after means it argues against a concrete number. Running it before
means it comments on a plan rather than on a conclusion, and gets ignored.

The bound is deliberate. An unbounded critic can turn any panel into a
single-number pipeline, which destroys the ensemble property that justified the
panel in the first place.

## 3. Resolution criteria are mandatory

`Question.resolution_criteria` is a required field, and the API rejects
questions without it.

When analysts answer slightly different questions, the spread between them
looks like uncertainty and the mean looks like an estimate. It is neither. In
a calibration study the single largest source of unusable forecasts was
ambiguity in what counted as "resolved", not a modelling error.

## 4. Confidence-weighted mean, trimmed when the panel is large enough

For four or more views the highest and lowest are dropped before averaging. One
analyst having a bad day should not move the headline number.

Weights are the analysts' self-reported confidence. This is a weak signal —
self-reported confidence is poorly calibrated in language models and humans
alike — but it is better than uniform weighting, and it is the first thing to
replace when per-persona weights can be learned from resolved history.

## 5. Interval width comes from disagreement

`half_width = min(0.35, max(0.05, 1.5 * disagreement + 0.06 + |shift|))`.

The 1.5 multiplier and the floor keep the interval informative when the panel
is tight (the 0.06 term) and prevent it from becoming useless when the panel
splits (the 0.35 cap). The red-team shift is included so a critic that moves
the number also widens the interval, which is the honest response to "we are
less sure than the panel thinks".

This is not a probabilistic interval. It is a disagreement interval and should
not be read as a 95% credible interval. Doing that properly needs a fitted
model, not a heuristic.

## 6. TF-IDF instead of embeddings

The corpus a forecasting panel reads is measured in hundreds of documents, not
millions. At that size the recall difference between lexical and dense
retrieval is small, while the difference in operational cost is large: no
index to build, no model to download, no network dependency.

The recency weighting is the part that actually matters. A document from four
months ago can still be structurally relevant, so stale documents stay
reachable at a discount (`0.4 + 0.6 * freshness`) rather than being cut off.

Swapping in a vector store means implementing one `search` method. The
retriever is passed in, not constructed inside the pipeline.

## 7. Recency half-life of 45 days

A half-life of two weeks throws away structural evidence too aggressively; three
months lets stale narratives dominate. 45 days sits where breaking news
dominates the top of the ranking but background documents remain reachable.

This is a guess, not a measurement. The right way to set it is to sweep the
half-life against a resolved history and pick the value that maximises
downstream accuracy. The parameter is exposed precisely so that sweep is
cheap to write.

## 8. Deterministic stub provider instead of mocking at the HTTP level

Tests would normally stub the model call at the transport boundary. This
project stubs it at the provider interface instead, with a deterministic lexical
model.

Two reasons. First, it gives the evaluation harness something to score, so the
calibration code path is exercised in CI rather than only in production.
Second, it means the whole system can be demonstrated without a key, which
makes review of the aggregation and calibration logic possible by anyone.

The stub's probabilities are deliberately imperfect. If it were perfect, the
reliability bins would be uninformative and the Platt scaling would look
magical. It is overconfident by construction, which is the failure mode real
forecasting systems actually have.

## 9. The split is shuffled, and the recalibration refuses degenerate fits

Recalibration fitted and evaluated on the same questions manufactures skill.
The evaluation harness shuffles with a fixed seed, splits 70/30, fits on the
training split, and reports the held-out split both before and after
recalibration.

The shuffle is not cosmetic. It was added after the harness produced a
recalibrated Brier score of 0.85 against a raw score of 0.24 on the same
questions. The cause was a dataset ordered by outcome: the sequential split
produced a training set containing only one outcome class, the intercept
learned there was unbounded, and the fitted curve extrapolated into the test
set with total confidence. A sequential split on an ordered file is a silent
landmine, which is why the split is now shuffled and the behaviour is covered
by a test.

The scaler is skipped entirely below eight training examples, and now also
when the training split contains a single outcome class. In both cases the
right answer is to ship the raw probability with an honest "we could not
calibrate this" rather than a confident-looking recalibration.

## 9a. Why the evaluation set is real events

The first version of this dataset was synthetic questions with outcomes I
assigned arbitrarily. It was worse than useless: it produced a Brier score
that looked like a measurement of the system and was actually a measurement of
how the outcomes happened to be ordered in the file.

The dataset is now 22 central bank decisions from 2024 and 2025, each with a
resolution rule stated in terms of the published decision, each with a
verifiable outcome. These are the kind of questions forecasting systems are
actually asked: binary, dated, publicly resolvable, with a reference class.

The offline stub still has no predictive power on them, and the reported
numbers say so. That is the correct outcome for a lexical model with no
knowledge of monetary policy. The harness, the split and the recalibration
path are what the dataset is there to exercise.

## 9b. What the calibration numbers do and do not show

Recalibration improves the held-out log loss from 0.704 to 0.653 and the
Brier score from 0.256 to 0.224. It does not beat the base rate Brier of
0.204, and it should not: a forecaster with no information cannot beat the
base rate no matter how well its probabilities are scaled. What the numbers
demonstrate is that the recalibration path is wired correctly and that the
split does not leak.

With 22 questions and a held-out split of 7, the standard error on a Brier
score is larger than the gap between any two rows in the table. Reading a
ranking into these numbers would be a mistake, and the harness reports them
because the point is to make the comparison possible once there are enough
questions to support it.

## 10. JSONL traces, SQLite for state

The audit trail is a JSONL file per run. It is append-only, human readable,
trivially greppable, and survives a crash mid-write without corrupting the
previous lines.

SQLite holds the forecasts, because they need to be queried, sorted and
retrieved by id. The two have different access patterns and forcing them into
one store would make one of them worse.

## 11. What is deliberately not here

- **Authentication and rate limiting on the API.** This is an internal tool
  shape. Adding auth without knowing the deployment topology is worse than
  leaving it out explicitly.
- **Async persistence.** SQLite writes are synchronous. They are fast enough
  for read-mostly internal use and would need a queue and a real database
  before they were not.
- **A vector index.** See section 6.
- **Learned analyst weights.** See section 4. This is the highest-value
  next step and it needs resolved history, which this repository does not yet
  have enough of.

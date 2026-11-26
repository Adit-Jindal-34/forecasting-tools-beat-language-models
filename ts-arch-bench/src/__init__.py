"""ts-arch-bench (corrected build) — core framework.

Version counters. These exist so that a change to one stage of the pipeline
invalidates exactly that stage's cached work and nothing else. The old build
had a single PROMPT_VERSION that invalidated everything, which made a
one-line parser fix as expensive as a full rerun.

  PROMPT_VERSION   bump when any prompt TEMPLATE changes (models/prompts.py).
                   -> invalidates cached MODEL CALLS (expensive).

  QUERY_VERSION    bump when the paraphrase bank changes (src/samples.py).
                   -> changes the sample pool, because the question text is
                      part of every sample. Enters data_fingerprint(), so a
                      reworded paraphrase regenerates the pool rather than
                      serving stale series under unchanged sample ids - the
                      same failure mode the seed-in-fingerprint fix prevents.
                      Also invalidates calls, since the question is in the
                      prompt.

  PARSER_VERSION   bump when routing/answer parsing changes.
                   -> invalidates nothing; systems are re-executed against
                      the warm call cache at zero API cost.

  SCORER_VERSION   bump when a metric changes.
                   -> invalidates nothing; scoring never calls a model.

Only PROMPT_VERSION and QUERY_VERSION cost money. That separation is the
whole point.
"""

__version__ = "1.1.0"

PROMPT_VERSION = 1
QUERY_VERSION = 1
PARSER_VERSION = 1
SCORER_VERSION = 1

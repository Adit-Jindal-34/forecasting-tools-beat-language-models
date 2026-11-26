"""Reader for the Monash `.tsf` format. numpy only - no scipy, no pandas.

The archive stores series in a Weka-ARFF derivative: a header of `@attribute`
/ `@frequency` / `@horizon` / `@missing` / `@equallength` lines, then one line
per series after `@data`:

    <series_name>:<start_timestamp>:v1,v2,v3,...

Values may be `?` when the dataset's `@missing` flag is true. This reader
returns them as `nan` and leaves the decision about them to the caller - the
pipeline assumes dense arrays, so a config should prefer a
`without_missing_values` variant rather than imputing silently.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Seasonal period used for the MASE denominator, per `@frequency`. These are
#: the archive's own conventions (Monash paper, Table 1), NOT estimated from
#: the data: the denominator must be a public, fixed quantity so it cannot
#: drift with the window the arms are shown. The arms still have to *estimate*
#: the period themselves - that stays part of the task.
FREQ_PERIOD: dict[str, int] = {
    "yearly": 1,
    "quarterly": 4,
    "monthly": 12,
    "weekly": 1,
    "daily": 7,
    "hourly": 24,
    "half_hourly": 48,
    "10_minutes": 144,
    "minutely": 60,
}


@dataclass(frozen=True)
class TsfFile:
    """One parsed `.tsf` file."""

    relation: str
    frequency: str
    horizon: int | None
    has_missing: bool
    equal_length: bool
    names: list[str]
    starts: list[str]
    series: list[np.ndarray]

    @property
    def period(self) -> int:
        return FREQ_PERIOD[self.frequency]

    def lengths(self) -> np.ndarray:
        return np.array([len(s) for s in self.series], dtype=int)


def read_tsf(path: str) -> TsfFile:
    relation, frequency, horizon = "", "", None
    has_missing = equal_length = False
    names: list[str] = []
    starts: list[str] = []
    series: list[np.ndarray] = []
    in_data = False

    # Several files carry cp1252 punctuation in their comment headers (an
    # en-dash in the M4 citation, for one), so strict utf-8 fails on the
    # header of a file whose data section is pure ASCII. Comments are
    # skipped, so replacing undecodable bytes is lossless for our purposes.
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if not in_data:
                low = line.lower()
                if low.startswith("@relation"):
                    relation = line.split(maxsplit=1)[1]
                elif low.startswith("@frequency"):
                    frequency = line.split(maxsplit=1)[1].lower()
                elif low.startswith("@horizon"):
                    horizon = int(line.split(maxsplit=1)[1])
                elif low.startswith("@missing"):
                    has_missing = line.split(maxsplit=1)[1].lower() == "true"
                elif low.startswith("@equallength"):
                    equal_length = line.split(maxsplit=1)[1].lower() == "true"
                elif low == "@data":
                    in_data = True
                continue

            # `name:start:v1,v2,...` - split on the first two colons only,
            # because the timestamp itself contains none in this format but
            # the series name may.
            head, start, values = line.split(":", 2)
            names.append(head)
            starts.append(start)
            series.append(np.array(
                [np.nan if v == "?" else float(v) for v in values.split(",")],
                dtype=float))

    if not frequency:
        raise ValueError(f"{path}: no @frequency line; cannot fix a period")
    return TsfFile(relation, frequency, horizon, has_missing, equal_length,
                   names, starts, series)

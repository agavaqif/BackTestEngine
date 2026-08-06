"""Strategy parameterisation (Functional requirement 3).

A strategy declares its tunable inputs once, as a :class:`ParameterSpec`. That
declaration buys three things the engine needs:

* **Validation before the run** — a bad parameter is a config error at second
  zero, not a ``TypeError`` on bar 40,000.
* **Reproducibility** — the resolved parameter set is hashed into the run id
  and stored on the result, so two runs are comparable only if they truly used
  the same inputs.
* **Sweepability** — :meth:`ParameterSpec.grid` enumerates the search space,
  which is the hook the future parameter-optimisation feature plugs into.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from sigmaloop.types import ParamDict, ParamValue

__all__ = ["Parameter", "ParameterSpec", "ParameterSet"]


@dataclass(frozen=True, slots=True)
class Parameter:
    """One declared strategy input.

    ``choices`` and ``(min_value, max_value)`` are alternative constraint
    styles; supplying both is a configuration error.

    ``sweep_values`` is what optimisation enumerates. When absent, the
    parameter is held at its default during a sweep — an explicit opt-in, so
    adding a parameter never silently multiplies the search space.
    """

    name: str
    default: ParamValue
    description: str = ""
    min_value: float | None = None
    max_value: float | None = None
    choices: tuple[ParamValue, ...] | None = None
    #: Python type the value is coerced to; inferred from ``default`` if None.
    value_type: type | None = None
    sweep_values: tuple[ParamValue, ...] | None = None

    def validate(self, value: ParamValue) -> ParamValue:
        """Coerce and range-check; raises ``ValidationError`` with the bounds."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """The full declared parameter surface of a strategy class."""

    parameters: tuple[Parameter, ...] = ()

    def __post_init__(self) -> None:
        """Reject duplicate names."""
        raise NotImplementedError

    def get(self, name: str) -> Parameter:
        raise NotImplementedError

    def defaults(self) -> ParamDict:
        raise NotImplementedError

    def resolve(self, overrides: Mapping[str, Any]) -> ParameterSet:
        """Merge overrides onto defaults, validating each and rejecting unknown
        names — a typo in a parameter name is a silent no-op in most engines and
        an error in this one."""
        raise NotImplementedError

    def grid(self) -> Iterator[ParamDict]:
        """Cartesian product over every parameter's ``sweep_values``."""
        raise NotImplementedError

    def grid_size(self) -> int:
        raise NotImplementedError

    @classmethod
    def from_parameters(cls, *parameters: Parameter) -> ParameterSpec:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ParameterSet:
    """Resolved, validated parameter values for one run.

    Exposes both mapping access (``params["period"]``) and attribute access
    (``params.period``) so strategy code reads naturally either way.
    """

    values: ParamDict = field(default_factory=dict)
    spec: ParameterSpec | None = None

    def __getitem__(self, name: str) -> ParamValue:
        raise NotImplementedError

    def __getattr__(self, name: str) -> ParamValue:
        raise NotImplementedError

    def __contains__(self, name: object) -> bool:
        raise NotImplementedError

    def get(self, name: str, default: ParamValue = None) -> ParamValue:
        raise NotImplementedError

    def as_dict(self) -> ParamDict:
        raise NotImplementedError

    def fingerprint(self) -> str:
        """Stable hash of the values — part of the run id and the cache key."""
        raise NotImplementedError

    def with_overrides(self, **overrides: ParamValue) -> ParameterSet:
        raise NotImplementedError

    def describe(self) -> Sequence[tuple[str, ParamValue, str]]:
        """``(name, value, description)`` rows for the human-readable summary."""
        raise NotImplementedError

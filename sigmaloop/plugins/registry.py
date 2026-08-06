"""Plugin registry (NFR 4).

Every swappable component — data providers, indicators, execution models,
spread/slippage/commission models, sizers, metric calculators, reporters — is
looked up by name through a registry rather than imported directly by the
engine. Two consequences:

* The engine depends on abstractions only. Nothing in
  :mod:`sigmaloop.engine` imports a concrete provider.
* Configs are plain strings (``"polygon"``, ``"next_bar_open"``), which makes
  them serialisable, diffable and hashable into the run fingerprint.

Registration happens two ways:

1. **Decorator** — ``@register(DATA_PROVIDERS)`` on a class in this codebase.
2. **Entry points** — third-party packages declare
   ``[project.entry-points."sigmaloop.data_providers"]``; these are discovered
   lazily on first lookup, so import cost is paid only if used.

Lookups fail loudly: an unknown name raises
:class:`~sigmaloop.errors.PluginNotFoundError` listing what *is* available,
because a silent fallback to a default provider would produce a plausible but
wrong backtest.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Generic, TypeVar

__all__ = [
    "PluginRegistry",
    "register",
    "DATA_PROVIDERS",
    "INDICATORS",
    "EXECUTION_MODELS",
    "SPREAD_MODELS",
    "SLIPPAGE_MODELS",
    "COMMISSION_MODELS",
    "POSITION_SIZERS",
    "RISK_CHECKS",
    "METRIC_CALCULATORS",
    "REPORTERS",
    "CALENDARS",
    "all_registries",
]

T = TypeVar("T")


class PluginRegistry(Generic[T]):
    """Name -> class mapping for one extension point.

    Entry-point discovery is lazy and memoised: the first :meth:`get` or
    :meth:`available` triggers a scan of the group, after which lookups are
    plain dict hits.
    """

    __slots__ = ("group", "_base", "_registered", "_entry_points_loaded")

    def __init__(self, group: str, base: type[T]) -> None:
        # Trivial by necessity: the module-level registries below are
        # constructed at import time, so this constructor must not raise.
        self.group = group
        self._base = base
        self._registered: dict[str, type[T]] = {}
        self._entry_points_loaded = False

    def register(self, name: str, plugin: type[T], *, override: bool = False) -> type[T]:
        """Register a class. Raises ``DuplicatePluginError`` unless ``override``."""
        raise NotImplementedError

    def get(self, name: str) -> type[T]:
        """Look up by name; raises ``PluginNotFoundError`` listing alternatives."""
        raise NotImplementedError

    def create(self, name: str, /, **kwargs: object) -> T:
        """Look up and instantiate.

        Constructor errors are re-raised as ``ConfigurationError`` naming the
        plugin and the offending option, since these almost always come from a
        bad config rather than a bug in the plugin.
        """
        raise NotImplementedError

    def available(self) -> Sequence[str]:
        """Every registered name, including entry-point plugins."""
        raise NotImplementedError

    def describe(self) -> Mapping[str, str]:
        """``name -> first docstring line``, for CLI help and error messages."""
        raise NotImplementedError

    def __contains__(self, name: object) -> bool:
        raise NotImplementedError

    def __iter__(self) -> Iterator[str]:
        raise NotImplementedError

    def _load_entry_points(self) -> None:
        """Scan ``self.group``. Failures are warnings, never fatal — one broken
        third-party plugin must not prevent the engine from starting."""
        raise NotImplementedError


def register(registry: PluginRegistry[T], name: str | None = None) -> Callable[[type[T]], type[T]]:
    """Class decorator. Uses the class's ``name`` attribute when ``name`` is None.

    ::

        @register(DATA_PROVIDERS)
        class MyProvider(DataProvider):
            name = "mine"
    """
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# The extension points.
#
# Declared with ``object`` as the base here to avoid import cycles (this module
# sits below every implementation package); the real base class is bound during
# ``sigmaloop.plugins.bootstrap``, which runs on first package import.
# --------------------------------------------------------------------------- #

DATA_PROVIDERS: PluginRegistry[object] = PluginRegistry("sigmaloop.data_providers", object)
INDICATORS: PluginRegistry[object] = PluginRegistry("sigmaloop.indicators", object)
EXECUTION_MODELS: PluginRegistry[object] = PluginRegistry("sigmaloop.execution_models", object)
SPREAD_MODELS: PluginRegistry[object] = PluginRegistry("sigmaloop.spread_models", object)
SLIPPAGE_MODELS: PluginRegistry[object] = PluginRegistry("sigmaloop.slippage_models", object)
COMMISSION_MODELS: PluginRegistry[object] = PluginRegistry("sigmaloop.commission_models", object)
POSITION_SIZERS: PluginRegistry[object] = PluginRegistry("sigmaloop.position_sizers", object)
RISK_CHECKS: PluginRegistry[object] = PluginRegistry("sigmaloop.risk_checks", object)
METRIC_CALCULATORS: PluginRegistry[object] = PluginRegistry("sigmaloop.metrics", object)
REPORTERS: PluginRegistry[object] = PluginRegistry("sigmaloop.reporters", object)
CALENDARS: PluginRegistry[object] = PluginRegistry("sigmaloop.calendars", object)


def all_registries() -> Mapping[str, PluginRegistry[object]]:
    """Every extension point, for ``sigmaloop plugins list``."""
    raise NotImplementedError

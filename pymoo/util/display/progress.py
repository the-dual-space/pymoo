"""Progress bar wrapper for optimization visualization."""

import math
from types import TracebackType
from typing import Any


class ProgressBar:
    """Progress bar for displaying optimization progress.

    Parameters:
        total: Total number of progress units.
        start: Whether to start the progress bar immediately.
        non_decreasing: Whether to ensure progress only increases.
        **kwargs: Keyword arguments passed to tqdm.
    """

    def __init__(self, total: float = 1.0, *, start: bool = True, non_decreasing: bool = True, **kwargs: Any) -> None:
        self.total = total
        kwargs.setdefault("bar_format", "{l_bar}{bar}| [{elapsed}<{remaining}]")
        self.kwargs = kwargs
        self.obj: Any = None
        self.non_decreasing = non_decreasing
        self._max = 0.0

        if start:
            self.start()

    def set(self, value: float, *args: Any, **kwargs: Any) -> None:
        """Set the progress bar value.

        Args:
            value: Progress value between 0 and 1.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        if self.non_decreasing:
            self._max = max(self._max, value)
            value = self._max

        prec = 100
        value = math.floor(value * prec) / prec

        self.obj.n = value * self.total
        self.obj.refresh(*args, **kwargs)

    def start(self) -> None:
        """Start the progress bar."""
        if self.obj is None:
            from tqdm.auto import tqdm

            self.obj = tqdm(total=self.total, **self.kwargs)

    def close(self) -> None:
        """Close the progress bar."""
        if self.obj is not None:
            self.obj.close()
            self.obj = None

    def __enter__(self) -> "ProgressBar":  # noqa: PYI034
        """Enter context manager."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit context manager."""
        self.close()

"""Tests for the tqdm-backed ProgressBar wrapper."""

from io import StringIO

import pytest

from pymoo.util.display.progress import ProgressBar


def test_progress_bar_updates_manually_and_never_decreases():
    progress = ProgressBar(total=100, start=False, file=StringIO(), mininterval=0)

    assert progress.obj is None
    progress.start()

    assert progress.obj.bar_format == "{l_bar}{bar}| [{elapsed}<{remaining}]"
    progress.set(0.429)
    assert progress.obj.n == pytest.approx(42.0)

    progress.set(0.2)
    assert progress.obj.n == pytest.approx(42.0)

    progress.set(1.0)
    assert progress.obj.n == pytest.approx(100.0)

    progress.close()
    assert progress.obj is None


def test_progress_bar_context_manager_starts_and_closes():
    progress = ProgressBar(start=False, file=StringIO(), bar_format="{n}/{total}")

    with progress as active:
        assert active is progress
        assert progress.obj is not None
        assert progress.obj.bar_format == "{n}/{total}"

    assert progress.obj is None


def test_progress_bar_can_decrease_when_requested():
    progress = ProgressBar(non_decreasing=False, file=StringIO(), mininterval=0)

    progress.set(0.8)
    progress.set(0.3)

    assert progress.obj.n == pytest.approx(0.3)
    progress.close()

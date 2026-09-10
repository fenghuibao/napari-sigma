"""Temporary files outlive test-local arrays and GUI objects on Windows."""
from contextlib import contextmanager
import gc
import tempfile


@contextmanager
def temporary_directory(test_case):
    directory = tempfile.TemporaryDirectory(prefix="sigma-test-")

    def cleanup():
        # Test locals must have gone out of scope and viewer.close() must have
        # run before collecting cycles and removing memory-mapped TIFF files.
        gc.collect()
        directory.cleanup()

    test_case.addCleanup(cleanup)
    yield directory.name

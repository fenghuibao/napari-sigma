"""Run real cold-font checks against installed assets in isolated processes."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class BundledFontTests(unittest.TestCase):
    def check(self, scenario):
        resources = Path(os.environ.get("SIGMA_DESKTOP_TEST_RESOURCES", Path(sys.prefix) / "sigma-desktop"))
        with tempfile.TemporaryDirectory(prefix="sigma-font-test-") as directory:
            result = subprocess.run(
                [sys.executable, "-I", "-B", str(Path(__file__).with_name("font_probe.py")),
                 str(resources), scenario, directory], capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            print(result.stdout, end="", flush=True)

    def test_fresh_cache_and_png_svg_pdf_rendering_without_discovery(self):
        self.check("fresh")

    def test_deleted_and_corrupt_user_cache_are_restored_without_discovery(self):
        self.check("cache-recovery")

    def test_relocated_app_has_no_build_machine_font_paths(self):
        self.check("relocated")

    def test_wrong_matplotlib_version_fails_without_discovery(self):
        self.check("wrong-version")

    def test_damaged_index_fails_without_discovery(self):
        self.check("damaged-index")

    def test_damaged_bundled_font_fails_without_discovery(self):
        self.check("damaged-font")


if __name__ == "__main__":
    unittest.main()

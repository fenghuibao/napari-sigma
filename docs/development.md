# Development

For environment setup, see [Development installation](../README.md#development-installation).

## Development checks

Run from the repository root with the development environment active:

```bash
python -m unittest discover -s tests -v
```

GUI tests require a display (on headless Linux, use `xvfb-run -a`). Release builds are gated on regression tests on macOS/Linux and Python 3.11/3.13. See [CHANGELOG.md](../CHANGELOG.md) for changes that affect measurements and saved data.

## Build validation

Build and validate the distribution locally from the repository root:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

## Release process

SIGMA is published as **`napari-sigma`**. The GitHub workflow uses PyPI Trusted Publishing; no long-lived PyPI API token is stored in the repository.

1. Update `src/napari_sigma/_version.py`.
2. Run the build validation commands above.
3. Commit the release changes.
4. Create and push a matching version tag, for example:

```bash
git tag v0.0.1
git push origin v0.0.1
```

The tag triggers distribution validation and publishes the built package to PyPI. The PyPI Trusted Publisher must be configured for this repository, the `test_and_deploy.yml` workflow, and the `pypi` environment.

## Reporting issues

Report bugs or share suggestions through [GitHub Issues](https://github.com/fenghuibao/napari-sigma/issues).

When reporting a problem, include:

- operating system and hardware
- Python, napari, Qt, and SIGMA versions
- computational device (`cpu`, `cuda`, or `mps`)
- input shape and axis interpretation
- traceback or warning output

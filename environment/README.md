# Pinned environment

Reference machine for every number in the paper:

| | |
|---|---|
| Chip | Apple M4 Pro |
| macOS | 26.6.2 (build 25G83) |
| Xcode | 26.2 (17C52) |
| Python | 3.12.12 |

`requirements.lock` is the exact `pip freeze` of the venv used. Recreate it with:

```bash
uv venv .venv-repro --python 3.12
uv pip install --python .venv-repro/bin/python -r environment/requirements.lock
uv pip install --python .venv-repro/bin/python --no-deps -e .
```

Note: `torch 2.10.0` is newer than the newest version coremltools 9.0 is tested
with (2.7.0); conversion works for the models in this repo but the warning is
expected. Results from a different macOS / chip must not be merged with these.

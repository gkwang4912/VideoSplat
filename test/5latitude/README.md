# Five-latitude Gaussian Camera Dataset

Place one Gaussian PLY and a complete COLMAP text sparse model under `input/`, then run:

```powershell
python -m pip install -r requirements.txt
python run.py
```

The run fails explicitly on missing, malformed, unsupported, or ambiguous input. It rebuilds only `output/`; it never writes to `input/`.

Change ring elevations, azimuth spacing, mask generation, render settings, and display-level splat settings in `config.json`. Validate an existing output independently with:

```powershell
python scripts/validate_project.py
```

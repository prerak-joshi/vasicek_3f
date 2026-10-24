# Contributing

Thanks for your interest. Issues and pull requests are welcome, especially new diagnostics, data
sources and model variants (e.g. survey-augmented P dynamics, shadow-rate extensions).

## Development setup

```bash
git clone https://github.com/prerak-joshi/vasicek_3f.git && cd vasicek_3f
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt && pip install -e ".[dev,docs]"
pre-commit install
```

## Before opening a pull request

```bash
ruff check . && ruff format --check .
pytest --cov
mkdocs build --strict          # if you changed docs
```

## Guidelines

- **All model logic lives in `src/vasicek3f/`.** The CLI, notebook and dashboard call the package;
  they should not re-implement pricing or estimation.
- **New model code needs tests**, ideally including a check against an independent calculation
  (closed form vs ODE, filter vs brute force) or a simulation-recovery test.
- Keep units consistent: yields in decimals internally, annualised time, percent/bp only for display.
- Document methodology changes in `docs/` and add a line to `CHANGELOG.md`.
- Regenerate the published results (`results/simulated`, and `results/gsw` if you have the data) when
  outputs change, using the commands in `docs/getting-started.md`.

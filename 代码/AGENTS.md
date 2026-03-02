# Repository Guidelines

## Project Structure & Module Organization
This repository is currently MATLAB-first and centered on a single script:

- `test.m`: main optimization and analysis workflow for LFM window-function design.

As the project grows, keep code split by responsibility:

- `src/`: reusable MATLAB functions (signal generation, metrics, constraints).
- `tests/`: verification scripts and regression checks.
- `assets/`: figures, reference data, and exported results.

Prefer moving helper functions out of monolithic scripts into `src/` once reused.

## Build, Test, and Development Commands
Run locally with MATLAB batch mode (recommended for reproducibility):

```powershell
matlab -batch "run('test.m')"
```

Optional GNU Octave run (if compatible):

```powershell
octave --quiet test.m
```

If you add test scripts under `tests/`, run them in batch as well, for example:

```powershell
matlab -batch "run('tests/test_metrics.m')"
```

## Coding Style & Naming Conventions
- Use 4-space indentation; no tabs.
- Keep one logical operation per line for optimization and constraint code.
- Use `snake_case` for local variables and helper functions (for example, `compute_pslr`, `mw_target_refined`).
- Use uppercase abbreviations for domain metrics (`PSLR`, `PAPR`, `LFM`) in comments and printed output.
- Add brief comments only where math/optimization intent is not obvious.

## Testing Guidelines
No formal framework is configured yet. Use script-based regression checks:

- Create test files as `tests/test_*.m`.
- Validate numerical behavior with fixed seeds (`rng(...)`) and explicit tolerances.
- Check key outputs: constraint satisfaction, metric deltas, and convergence status.
- For algorithm changes, include at least one before/after metric comparison in the test output.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit messages (for example, `Fix fmincon barrier failure...`, `Improve PSLR gap...`).

- Commit format: `<Verb> <scope> <impact>`.
- Keep commits focused; avoid mixing refactors with algorithm changes.
- PRs should include:
  - what changed and why,
  - affected metrics (PSLR/MW/PAPR),
  - how to reproduce (`matlab -batch ...`),
  - plots/screenshots when visualization output changes.

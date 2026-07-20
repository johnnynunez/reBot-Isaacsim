# Versioned validation baselines

These four JSON files are required inputs to `scripts/make_validation_md.py`.
They contain the July 2026 gain-tuner 3.5.3 baseline measurements used by the
existing comparison tables in `VALIDATION.md`:

- `gt_pj_newasset_newton.json`
- `gt_pj_newasset_physx.json`
- `gt_grip_decomp_newton.json`
- `gt_grip_decomp_physx.json`

They were previously read from an absolute path in the author's home directory.
Versioning the exact numeric inputs makes Markdown regeneration deterministic on
a clean checkout and makes missing evidence a hard failure instead of silently
emitting empty comparison cells.

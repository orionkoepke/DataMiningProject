# tests

This folder holds **all tests** for the code in **lib**.

---

## Subdirectories

| Directory        | Purpose |
|------------------|--------|
| **`unit/`**      | Unit tests. Must be fast and must not rely on outside entities (no network, no broker, no real APIs). |
| **`integration/`** | Integration tests. Can be slower and may rely on outside entities (e.g. pulling real data from a stock broker). |

Run the full suite with `bin/ctl.py test` or `bin/ctl.bat test` from the project root. Use `--skip-unit` or `--skip-integration` to run only one kind.

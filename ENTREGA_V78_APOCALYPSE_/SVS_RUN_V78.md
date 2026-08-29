# SVS V1 run against POLYDIM V78

- Total tests: **39**
- Passed: **28**
- Failed: **11**
- Errors: **0**
- Skipped: **0**

Command:

```bash
python -m pytest -q scientific_suite/tests --tb=short
```

The suite intentionally treats specification violations as failures rather than `xfail`.

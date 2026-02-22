# Prompt for Codex: Fix CATEGORY and _AR/_EN Column Consistency

## Context

Use **only** `Schema.csv` and `Metric Indicators.csv` as the source of truth. Fix all discrepancies in the codebase (e.g. `generate_data.py`, `Schema.sql`, `generated_data_test.sql`, `generated_data_test_fixed.sql`, `index.html` if it contains embedded SQL) so that:

1. The `gac_fact_time_to_import_and_export` table has correct category values.
2. Every `*_AR` column contains proper Arabic translations of its corresponding `*_EN` column.

---

## Task 1: Fix `gac_fact_time_to_import_and_export` Category Values

For the table `gac_fact_time_to_import_and_export` (Indicator 37: Time to Export, Indicator 38: Time to Import):

- **CATEGORY_EN** must use exactly these two values: `"Export"` and `"Import"`.
- **CATEGORY_AR** must be the Arabic translation of the `_EN` value:
  - `"Export"` → `"تصدير"`
  - `"Import"` → `"استيراد"`

Update:
1. **generate_data.py**: Add table-specific logic so that when generating rows for `gac_fact_time_to_import_and_export`, `CATEGORY_EN` is chosen from `["Export", "Import"]` and `CATEGORY_AR` is set to the matching Arabic value.
2. **All SQL files** (`generated_data_test.sql`, `generated_data_test_fixed.sql`, `Schema.sql` sample data if any, and any embedded SQL in `index.html`): Replace any existing `gac_fact_time_to_import_and_export` INSERT values for `CATEGORY_EN` and `CATEGORY_AR` with the correct `Export`/`تصدير` and `Import`/`استيراد` pairs.

---

## Task 2: Enforce _AR = Arabic Translation of _EN Everywhere

For every table and every column pair where a column `X_EN` has a matching `X_AR`:

- **Rule**: `X_AR` must contain the correct Arabic translation of the value in `X_EN`.
- Use `Metric Indicators.csv` column "Short Description AR" where indicator names or descriptions are involved.
- For common terms, use standard Arabic translations, e.g.:
  - Export → تصدير
  - Import → استيراد
  - General → عام
  - Services → خدمات
  - Industry → صناعة
  - Trade → تجارة
  - Public → عام
  - Private → خاص

**In `generate_data.py`**:
- Replace the generic `_AR` handling that returns fake words with logic that, for each `X_EN`/`X_AR` pair, sets `X_AR` to the proper Arabic translation of `X_EN`.
- Add table- and column-specific mappings where needed (e.g. for `gac_fact_time_to_import_and_export` as above).
- Ensure any `CATEGORY`, `TYPE`, `GROUP`, or `SECTOR` columns that have `_EN`/`_AR` pairs follow this rule.

**In SQL data files**:
- Scan all INSERT statements for tables that have `*_AR` columns.
- Ensure every `*_AR` value is a valid Arabic translation of the corresponding `*_EN` value in the same row.
- Fix or regenerate rows that violate this rule.

---

## Deliverables

1. Updated `generate_data.py` with correct category handling for `gac_fact_time_to_import_and_export` and proper `_AR`/`_EN` translation logic.
2. Updated SQL files so that `gac_fact_time_to_import_and_export` uses `Export`/`تصدير` and `Import`/`استيراد` only.
3. All other `*_AR` values aligned with their `*_EN` translations according to `Metric Indicators.csv` and `Schema.csv`.

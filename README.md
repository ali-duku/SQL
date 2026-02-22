# Duplexer Sport SQL Explorer

`index.html` is a single-file, browser-based SQL explorer for the embedded `gold` dataset.
It runs fully client-side and is designed to work on GitHub Pages (no backend required).

## What It Does

- Loads an embedded SQL dump from inside `index.html` (`<script id="seedSql">`).
- Creates an in-memory SQLite database in the browser using `sql.js`.
- Uses schema `gold` so queries can be written as `gold.table_name`.
- Enforces **read-only SQL** (querying only).
- Shows:
  - table list with search by table name and column name,
  - focused schema relationship visualization,
  - query editor with autocomplete,
  - result grid with sticky headers and Sheets-like sort/filter menus.

## Main Features

### 1) Read-only query runner

- Allowed query starts: `SELECT`, `WITH`, `PRAGMA`
- Write/DDL statements are blocked (e.g. `INSERT`, `UPDATE`, `DELETE`, `CREATE`, `DROP`, etc.).
- `Ctrl+Enter` (or `Cmd+Enter`) executes the current query.

### 2) SQL autocomplete

Autocomplete in the query editor includes:

- SQL keywords and syntax snippets,
- SQL functions,
- table names,
- column names,
- alias-aware suggestions for `alias.column`.

Keyboard support:

- `ArrowUp` / `ArrowDown` to navigate
- `Enter` / `Tab` to apply suggestion
- `Esc` to close suggestions

### 3) Schema visualization (focused)

- Visualization auto-focuses on the selected table.
- When running a query, it tries to infer the table from the first `SELECT` expression and refocuses automatically.
- Handles cases like:
  - `SELECT * FROM gold.dim_date ...`
  - `SELECT d.* FROM gold.dim_date d ...`
  - qualified first columns like `d.DATE_ID`.

### 4) Result table sorting and filtering

Each result column header has a small menu button with:

- `Sort A to Z`
- `Sort Z to A`
- `Filter by values` (checkbox list, search, select all/clear, OK/Cancel)

Headers stay sticky while scrolling result rows.

### 5) App-level scrolling behavior

- Whole page is fixed to viewport (`html/body` non-scrollable).
- Left table list is scrollable (search inputs stay fixed above it).
- Result panel is scrollable independently.

## Files

- `index.html`: production app (single-file UI + embedded SQL data)
- `sql_explorer.html`: alternate/working copy variant (if present)
- `generated_data_test_fixed.sql`: source SQL dump used/embedded in app
- `generate_data.py`: data generation utility (separate from the explorer runtime)

## Run Locally

Open `index.html` directly in a browser.

Notes:

- Internet access is needed for CDN-loaded `sql.js` (`cdnjs`).
- First load may take time due to embedded dataset size.

## Deploy to GitHub Pages

1. Commit `index.html` to your repo.
2. In repository settings, enable GitHub Pages for your branch/folder.
3. Use the published Pages URL.

No server-side runtime is required.

## Updating Embedded Data

If you regenerate data and want the explorer to use it:

1. Generate/update your SQL dump file.
2. Replace the content inside `index.html` under:
   - `<script id="seedSql" type="text/plain"> ... </script>`
3. Redeploy.

## Limitations

- Query engine is SQLite (`sql.js`), not MySQL runtime behavior in all edge cases.
- Very large datasets can increase initial load time and browser memory usage.
- Relationship graph is heuristic-based from column naming patterns (not explicit FK constraints).

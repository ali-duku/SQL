
#!/usr/bin/env python3
"""
MySQL 5.7-first synthetic data generator for the schema in Schema.sql.

Features:
- Creates database/tables from Schema.sql
- Reads Schema.csv and Metric Indicators.csv for metadata/semantics
- Generates deterministic high-volume synthetic data
- Infers FK-like relationships from ID columns and load order
- Supports batched inserts and optional CSV + LOAD DATA LOCAL INFILE
- Validates counts, inferred FK integrity, and basic value bounds
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import math
import numbers
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from faker import Faker

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


@dataclass
class ColumnMeta:
    table_name: str
    column_name: str
    data_type: str
    ordinal_position: int


@dataclass
class TableMeta:
    name: str
    columns: List[ColumnMeta]

    @property
    def column_names(self) -> List[str]:
        return [c.column_name for c in sorted(self.columns, key=lambda x: x.ordinal_position)]

    def dtype(self, column: str) -> str:
        for col in self.columns:
            if col.column_name == column:
                return col.data_type.upper()
        return "STRING"


def qident(name: str) -> str:
    return f"`{name.replace('`', '``')}`"


def split_sql_statements(sql_text: str) -> List[str]:
    parts = [p.strip() for p in sql_text.split(";")]
    return [p for p in parts if p]


def parse_schema_table_names(sql_path: Path) -> List[str]:
    sql_text = sql_path.read_text(encoding="utf-8")
    names = re.findall(r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+`([^`]+)`", sql_text, flags=re.I)
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def read_schema_csv(path: Path) -> Dict[str, TableMeta]:
    tables: Dict[str, List[ColumnMeta]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tbl = row["table_name"].strip()
            col = ColumnMeta(
                table_name=tbl,
                column_name=row["column_name"].strip(),
                data_type=row["data_type"].strip().upper(),
                ordinal_position=int(row.get("ordinal_position", 0)),
            )
            tables[tbl].append(col)
    return {name: TableMeta(name=name, columns=cols) for name, cols in tables.items()}


def map_csv_tables_to_schema(csv_tables: Sequence[str], schema_tables: Sequence[str]) -> Dict[str, str]:
    schema_by_lower = {t.lower(): t for t in schema_tables}
    mapping: Dict[str, str] = {}
    for csv_t in csv_tables:
        low = csv_t.lower()
        if low in schema_by_lower:
            mapping[csv_t] = schema_by_lower[low]
            continue
        candidates = [s for s in schema_tables if low.startswith(s.lower()) or s.lower().startswith(low)]
        if len(candidates) == 1:
            mapping[csv_t] = candidates[0]
            continue
        # fallback: remove vowels + non alnum to match aggressive truncation
        norm = re.sub(r"[^a-z0-9]", "", low)
        def strip_vowels(x: str) -> str:
            return re.sub(r"[aeiou]", "", re.sub(r"[^a-z0-9]", "", x.lower()))
        sv = strip_vowels(csv_t)
        candidates = [s for s in schema_tables if strip_vowels(s) == sv or strip_vowels(s) in sv or sv in strip_vowels(s)]
        if len(candidates) == 1:
            mapping[csv_t] = candidates[0]
    return mapping


def remap_table_metadata(raw_tables: Dict[str, TableMeta], name_map: Dict[str, str]) -> Dict[str, TableMeta]:
    remapped: Dict[str, TableMeta] = {}
    for old_name, tmeta in raw_tables.items():
        if old_name not in name_map:
            continue
        new_name = name_map[old_name]
        cols = [
            ColumnMeta(
                table_name=new_name,
                column_name=c.column_name,
                data_type=c.data_type,
                ordinal_position=c.ordinal_position,
            )
            for c in tmeta.columns
        ]
        remapped[new_name] = TableMeta(name=new_name, columns=cols)
    return remapped


def read_metric_map(path: Path, schema_tables: Sequence[str]) -> Dict[str, List[dict]]:
    schema_set = {t.lower() for t in schema_tables}
    mapping: Dict[str, List[dict]] = defaultdict(list)
    skipped = 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = (row.get("Gold Schema Table Name") or "").strip().lower()
            if not raw:
                continue
            table = raw.replace("gold.", "")
            if table in schema_set:
                mapping[table].append(row)
            else:
                skipped += 1
    logging.info("Metric mappings applied: %d tables, skipped references not in schema: %d", len(mapping), skipped)
    return mapping


def classify_table_role(table_name: str) -> str:
    t = table_name.lower()
    if t.startswith("dim_") or "_dim_" in t:
        return "dimension"
    if "_fact_" in t:
        return "fact"
    return "dimension" if t.startswith("dim") else "fact"


def detect_key_column(table: TableMeta) -> Optional[str]:
    cols = table.column_names
    for c in cols:
        if c.endswith("_ID") and (c.startswith("GLD_") or c == f"{table.name.upper()}_ID"):
            return c
    for c in cols:
        if c.endswith("_ID"):
            return c
    return cols[0] if cols else None


def infer_dimension_keys(tables: Dict[str, TableMeta]) -> Dict[str, Tuple[str, str]]:
    keys: Dict[str, Tuple[str, str]] = {}
    for tname, tmeta in tables.items():
        if classify_table_role(tname) != "dimension":
            continue
        key_col = detect_key_column(tmeta)
        if key_col:
            keys[key_col] = (tname, key_col)
    return keys


def infer_fk_like_edges(tables: Dict[str, TableMeta], dim_keys: Dict[str, Tuple[str, str]]) -> Dict[str, List[Tuple[str, str, str]]]:
    edges: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for tname, tmeta in tables.items():
        if classify_table_role(tname) == "dimension":
            continue
        key_col = detect_key_column(tmeta)
        for col in tmeta.column_names:
            if not col.endswith("_ID"):
                continue
            if col == key_col:
                continue
            if col in dim_keys:
                parent_table, parent_col = dim_keys[col]
                edges[tname].append((col, parent_table, parent_col))
    return edges


def topo_load_order(tables: Dict[str, TableMeta], fk_edges: Dict[str, List[Tuple[str, str, str]]]) -> List[str]:
    deps: Dict[str, set] = {t: set() for t in tables}
    for child, refs in fk_edges.items():
        for _, parent, _ in refs:
            if child != parent:
                deps[child].add(parent)
    ordered: List[str] = []
    remaining = set(tables.keys())
    while remaining:
        ready = sorted([t for t in remaining if not deps[t] - set(ordered)])
        if not ready:
            ordered.extend(sorted(remaining))
            break
        ordered.extend(ready)
        remaining -= set(ready)
    return ordered


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic MySQL 5.7 data for Schema.sql")
    p.add_argument("--mysql-host", default="127.0.0.1")
    p.add_argument("--mysql-port", type=int, default=3306)
    p.add_argument("--mysql-user")
    p.add_argument("--mysql-password", default=os.getenv("MYSQL_PASSWORD"))
    p.add_argument("--mysql-db", default="gold")
    p.add_argument("--schema-sql", default="Schema.sql")
    p.add_argument("--schema-csv", default="Schema.csv")
    p.add_argument("--metrics-csv", default="Metric Indicators.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fact-rows", type=int, default=2_000_000)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--start-year", type=int, default=2000)
    p.add_argument("--end-year", type=int, default=2026)
    p.add_argument("--batch-size", type=int, default=10_000)
    p.add_argument("--emit-csv", default=None)
    p.add_argument("--use-load-data", action="store_true")
    p.add_argument("--null-rate-min", type=float, default=0.02)
    p.add_argument("--null-rate-max", type=float, default=0.08)
    p.add_argument("--outlier-rate", type=float, default=0.01)
    p.add_argument("--skip-validation", action="store_true")
    p.add_argument("--output-sql", help="Write generated data SQL to this file instead of inserting directly")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def connect_mysql(args: argparse.Namespace):
    try:
        import mysql.connector  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("mysql-connector-python is required. Install with: pip install mysql-connector-python") from exc

    if not args.mysql_user:
        raise ValueError("MySQL user is required unless --output-sql is used")
    if not args.mysql_password:
        raise ValueError("MySQL password is required via --mysql-password or MYSQL_PASSWORD env var")

    conn = mysql.connector.connect(
        host=args.mysql_host,
        port=args.mysql_port,
        user=args.mysql_user,
        password=args.mysql_password,
        autocommit=False,
        allow_local_infile=True,
    )
    return conn


def execute_schema(conn, sql_path: Path, db_name: str) -> None:
    sql_text = sql_path.read_text(encoding="utf-8")
    sql_text = re.sub(r"CREATE\s+DATABASE\s+IF\s+NOT\s+EXISTS\s+`?\w+`?", f"CREATE DATABASE IF NOT EXISTS `{db_name}`", sql_text, flags=re.I)
    sql_text = re.sub(r"USE\s+`?\w+`?", f"USE `{db_name}`", sql_text, flags=re.I)
    cur = conn.cursor()
    for stmt in split_sql_statements(sql_text):
        cur.execute(stmt)
    conn.commit()
    cur.close()


def table_year_column(cols: Sequence[str]) -> Optional[str]:
    for candidate in ("YEAR", "SURVEY_YEAR"):
        if candidate in cols:
            return candidate
    return None


COUNTRY_DATA = [
    (634, "QAT", "Qatar", "Qatari", "QAR", "Qatari Riyal", "Middle East", 1),
    (784, "ARE", "United Arab Emirates", "Emirati", "AED", "UAE Dirham", "Middle East", 1),
    (682, "SAU", "Saudi Arabia", "Saudi", "SAR", "Saudi Riyal", "Middle East", 1),
    (414, "KWT", "Kuwait", "Kuwaiti", "KWD", "Kuwaiti Dinar", "Middle East", 1),
    (512, "OMN", "Oman", "Omani", "OMR", "Omani Rial", "Middle East", 1),
    (48, "BHR", "Bahrain", "Bahraini", "BHD", "Bahraini Dinar", "Middle East", 1),
    (840, "USA", "United States", "American", "USD", "US Dollar", "North America", 0),
    (826, "GBR", "United Kingdom", "British", "GBP", "Pound Sterling", "Europe", 0),
    (276, "DEU", "Germany", "German", "EUR", "Euro", "Europe", 0),
    (250, "FRA", "France", "French", "EUR", "Euro", "Europe", 0),
    (156, "CHN", "China", "Chinese", "CNY", "Yuan", "Asia", 0),
    (392, "JPN", "Japan", "Japanese", "JPY", "Yen", "Asia", 0),
    (356, "IND", "India", "Indian", "INR", "Indian Rupee", "Asia", 0),
    (410, "KOR", "South Korea", "Korean", "KRW", "Won", "Asia", 0),
    (36, "AUS", "Australia", "Australian", "AUD", "Australian Dollar", "Oceania", 0),
    (124, "CAN", "Canada", "Canadian", "CAD", "Canadian Dollar", "North America", 0),
    (724, "ESP", "Spain", "Spanish", "EUR", "Euro", "Europe", 0),
    (380, "ITA", "Italy", "Italian", "EUR", "Euro", "Europe", 0),
    (528, "NLD", "Netherlands", "Dutch", "EUR", "Euro", "Europe", 0),
    (752, "SWE", "Sweden", "Swedish", "SEK", "Swedish Krona", "Europe", 0),
]

class SyntheticGenerator:
    def __init__(self, args: argparse.Namespace, tables: Dict[str, TableMeta], metric_map: Dict[str, List[dict]]):
        self.args = args
        self.tables = tables
        self.metric_map = metric_map
        self.fake = Faker()
        Faker.seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        self.id_pools: Dict[str, List] = defaultdict(list)
        self.country_strength: Dict[int, float] = {}

        self.null_rates: Dict[Tuple[str, str], float] = {}
        for tname, tmeta in tables.items():
            for c in tmeta.column_names:
                self.null_rates[(tname, c)] = random.uniform(args.null_rate_min, args.null_rate_max)

    def allocate_rows(self) -> Dict[str, int]:
        dims = [t for t in self.tables if classify_table_role(t) == "dimension"]
        facts = [t for t in self.tables if classify_table_role(t) == "fact"]

        allocation: Dict[str, int] = {}
        for t in dims:
            if t == "dim_date":
                allocation[t] = 0
            elif t == "dim_country":
                allocation[t] = int(len(COUNTRY_DATA) * self.args.scale)
            elif t == "dim_indicators":
                allocation[t] = max(120, int(200 * self.args.scale))
            else:
                allocation[t] = max(3, int(20 * self.args.scale))

        widths = np.array([len(self.tables[t].column_names) for t in facts], dtype=float)
        weights = widths / widths.sum()
        raw = np.floor(weights * self.args.fact_rows).astype(int)
        for t, n in zip(facts, raw):
            allocation[t] = max(100, int(n))

        current_total = sum(allocation[t] for t in facts)
        diff = self.args.fact_rows - current_total
        idx = 0
        while diff != 0 and facts:
            t = facts[idx % len(facts)]
            step = 1 if diff > 0 else -1
            if allocation[t] + step >= 10:
                allocation[t] += step
                diff -= step
            idx += 1

        return allocation

    def make_date_rows(self) -> List[Tuple]:
        rows = []
        start = dt.date(self.args.start_year, 1, 1)
        end = dt.date(self.args.end_year, 12, 31)
        d = start
        while d <= end:
            date_id = int(d.strftime("%Y%m%d"))
            month_start = d.replace(day=1)
            if d.month == 12:
                next_month = d.replace(year=d.year + 1, month=1, day=1)
            else:
                next_month = d.replace(month=d.month + 1, day=1)
            month_end = next_month - dt.timedelta(days=1)
            quarter = (d.month - 1) // 3 + 1
            rows.append(
                (
                    date_id,
                    d,
                    int(d.strftime("%Y%m%d")),
                    d.month,
                    d.strftime("%B"),
                    d.strftime("%B"),
                    f"Q{quarter}",
                    f"Q{quarter}",
                    quarter,
                    f"Q{quarter}-{d.year}",
                    (d.day - 1) // 7 + 1,
                    int(d.strftime("%U")),
                    d.strftime("%Y-%m"),
                    month_start,
                    month_end,
                    d.year,
                )
            )
            d += dt.timedelta(days=1)
        self.id_pools["DATE_ID"] = [r[0] for r in rows]
        return rows

    def generate_dimension_rows(self, table: str, count: int) -> List[Tuple]:
        cols = self.tables[table].column_names
        if table == "dim_date":
            return self.make_date_rows()
        if table == "dim_country":
            rows = []
            for item in COUNTRY_DATA[: max(1, count)]:
                cid, iso, name, nat, ccode, ccy, region, gcc = item
                rows.append((cid, iso, name, nat, ccode, ccy, region, bool(gcc), f"https://flags.example/{iso.lower()}.png", name, nat, region))
            self.id_pools["COUNTRY_ID"] = [r[0] for r in rows]
            self.country_strength = {cid: np.clip(np.random.normal(1.0 + (0.3 if cid == 634 else 0), 0.2), 0.5, 1.7) for cid in self.id_pools["COUNTRY_ID"]}
            return rows

        if table == "dim_gender":
            rows = [
                (1, 1, 1, 1, "Male", "Male", "run-1", "trg-1", dt.datetime.utcnow(), "", "pipeline", "2026"),
                (2, 2, 2, 2, "Female", "Female", "run-1", "trg-1", dt.datetime.utcnow(), "", "pipeline", "2026"),
            ]
            self.id_pools["GENDER_ID"] = [1, 2]
            return [tuple(r[: len(cols)]) for r in rows]

        if table == "dim_marital_status":
            rows = [(1, "Single", "Single"), (2, "Married", "Married"), (3, "Divorced", "Divorced")]
            self.id_pools["MARITAL_STATUS_ID"] = [r[0] for r in rows]
            return rows

        if table == "hmc_dim_visit_type":
            rows = [(1, "Outpatient", "Outpatient"), (2, "Emergency", "Emergency"), (3, "Inpatient", "Inpatient")]
            self.id_pools["VISIT_TYPE_ID"] = [r[0] for r in rows]
            return rows

        if table == "dim_indicators":
            rows = []
            metrics = []
            for tbl_metrics in self.metric_map.values():
                metrics.extend(tbl_metrics)
            if not metrics:
                metrics = [{}] * count
            for i in range(max(count, len(metrics))):
                m = metrics[i % len(metrics)]
                iid = i + 1
                name = (m.get("Indicator SCEAI Name") if isinstance(m, dict) else None) or f"Indicator {i+1}"
                sector = (m.get("Indicator Sector (cleansed)") if isinstance(m, dict) else None) or random.choice(["Economic Diversification", "Human Capital", "Trade", "Health"])
                definition = (m.get("Short description EN") if isinstance(m, dict) else None) or f"Synthetic definition for {name}"
                row = [None] * len(cols)
                for j, c in enumerate(cols):
                    dtype = self.tables[table].dtype(c)
                    if c == "INDICATOR_ID":
                        row[j] = iid
                    elif c == "INDICATOR_NAME":
                        row[j] = name
                    elif c == "INDICATOR_SECTOR":
                        row[j] = sector
                    elif c == "INDICATOR_DEFINITION":
                        row[j] = definition
                    elif c == "EARLIEST_DATE_WHERE_QATAR_IS_AVAILABLE":
                        row[j] = self.args.start_year
                    elif c == "TARGET_VALUE_NUMERIC":
                        row[j] = round(np.random.uniform(10, 90), 3)
                    elif c == "LAST_UPDATED":
                        row[j] = dt.datetime.utcnow()
                    elif c == "TARGET_YEAR":
                        row[j] = self.args.end_year
                    elif c.endswith("_AR"):
                        row[j] = row[j - 1] if j > 0 else name
                    elif c in ("IS_QPULSE",):
                        row[j] = bool(np.random.rand() > 0.5)
                    elif dtype in ("LONG", "INT"):
                        row[j] = int(max(0, round(np.random.normal(50, 20))))
                    elif dtype in ("DOUBLE", "FLOAT", "DECIMAL"):
                        row[j] = round(float(np.random.uniform(0, 100)), 4)
                    else:
                        row[j] = f"{c.title()} {i+1}"
                rows.append(tuple(row))
            self.id_pools["INDICATOR_ID"] = [r[cols.index("INDICATOR_ID")] for r in rows if "INDICATOR_ID" in cols]
            return rows

        rows = []
        key_col = detect_key_column(self.tables[table])
        for i in range(1, count + 1):
            row = []
            for c in cols:
                row.append(self.make_value(table, c, i, {}))
            rows.append(tuple(row))
        if key_col:
            key_idx = cols.index(key_col)
            self.id_pools[key_col] = [r[key_idx] for r in rows]
        return rows

    def pick_year(self) -> int:
        return int(np.random.randint(self.args.start_year, self.args.end_year + 1))

    def should_null(self, table: str, col: str, is_key_like: bool) -> bool:
        if is_key_like:
            return False
        return np.random.rand() < self.null_rates[(table, col)]

    def numeric_value(self, col: str, country_id: Optional[int], year: Optional[int]) -> float:
        cname = col.upper()
        base = 1.0
        if country_id is not None:
            base *= float(self.country_strength.get(country_id, 1.0))
        if year is not None:
            base *= 1.0 + 0.015 * (year - self.args.start_year)

        if "RANK" in cname:
            return float(int(np.clip(np.random.normal(45 / base, 25), 1, 200)))
        if "RATIO" in cname or "PERCENT" in cname:
            return float(np.clip(np.random.normal(52 * base, 18), 0, 100))
        if "SCORE" in cname or "INDEX" in cname:
            return float(np.clip(np.random.normal(55 * base, 15), 0, 100))
        if any(k in cname for k in ["VALUE", "GDP", "REVENUE", "PRICE", "COST", "EXPENDITURE", "CAPITAL", "BUDGET", "TRADE", "FUNDING", "DEBT"]):
            return float(np.random.lognormal(mean=10.0 + math.log(base), sigma=0.55))
        if any(k in cname for k in ["TOTAL", "NUMBER", "COUNT", "DEALS", "VISITORS", "STUDENTS", "TEACHERS"]):
            return float(max(0, np.random.poisson(lam=1200 * base)))
        return float(np.random.lognormal(mean=6.0 + math.log(base), sigma=0.6))

    def make_value(self, table: str, col: str, idx: int, row_ctx: Dict[str, object]):
        dtype = self.tables[table].dtype(col)
        cname = col.upper()

        if cname in row_ctx:
            return row_ctx[cname]

        if cname.endswith("_ID") and cname in self.id_pools and not cname.startswith(("GLD_", "SLVR_", "BRNZ_")):
            vals = self.id_pools[cname]
            if vals:
                return random.choice(vals)

        if cname.startswith(("GLD_", "SLVR_", "BRNZ_")) and cname.endswith("_ID"):
            return idx
        if cname.endswith("_ID") and cname not in self.id_pools:
            return idx

        if cname in ("YEAR", "SURVEY_YEAR"):
            y = self.pick_year()
            row_ctx[cname] = y
            return y

        if cname == "DATE_ID":
            vals = self.id_pools.get("DATE_ID", [])
            if vals:
                date_id = random.choice(vals)
                row_ctx[cname] = date_id
                row_ctx["YEAR"] = int(str(date_id)[:4])
                return date_id

        if dtype in ("LONG", "INT"):
            if "RANK" in cname:
                return int(np.clip(np.random.normal(60, 30), 1, 250))
            if "YEAR" in cname:
                return self.pick_year()
            v = self.numeric_value(cname, row_ctx.get("COUNTRY_ID"), row_ctx.get("YEAR"))
            return int(max(0, round(v)))

        if dtype in ("DOUBLE", "FLOAT", "DECIMAL"):
            v = self.numeric_value(cname, row_ctx.get("COUNTRY_ID"), row_ctx.get("YEAR"))
            if np.random.rand() < self.args.outlier_rate:
                v *= np.random.uniform(3.0, 8.0)
            if "RATIO" in cname or "PERCENT" in cname or "SCORE" in cname or "INDEX" in cname:
                v = float(np.clip(v, 0, 100))
            return round(float(v), 4)

        if dtype == "BOOLEAN":
            return bool(np.random.rand() > 0.5)

        if dtype == "DATE":
            vals = self.id_pools.get("DATE_ID", [])
            if vals:
                date_id = random.choice(vals)
                return dt.datetime.strptime(str(date_id), "%Y%m%d").date()
            y = self.pick_year()
            m = int(np.random.randint(1, 13))
            d = int(np.random.randint(1, 28))
            return dt.date(y, m, d)

        if dtype == "TIMESTAMP":
            y = self.pick_year()
            m = int(np.random.randint(1, 13))
            d = int(np.random.randint(1, 28))
            hh = int(np.random.randint(0, 24))
            mm = int(np.random.randint(0, 60))
            ss = int(np.random.randint(0, 60))
            return dt.datetime(y, m, d, hh, mm, ss)

        if cname == "QID":
            return "".join(np.random.choice(list("0123456789"), size=11))
        if "ISO" in cname:
            return random.choice(["QAT", "USA", "DEU", "GBR", "CHN", "IND"])
        if "COUNTRY" in cname and "NAME" in cname:
            return random.choice(["Qatar", "Saudi Arabia", "United Arab Emirates", "United States", "Germany"])
        if "NATIONALITY" in cname:
            return random.choice(["Qatari", "Saudi", "Emirati", "American", "German"])
        if "CURRENCY_CODE" in cname:
            return random.choice(["QAR", "USD", "EUR", "SAR", "AED"])
        if "TREND_DIRECTION" in cname:
            return random.choice(["Up", "Down", "Stable"])
        if "CATEGORY" in cname or "TYPE" in cname or "GROUP" in cname or "SECTOR" in cname:
            return random.choice(["General", "Services", "Industry", "Trade", "Public", "Private"])
        if cname.endswith("_AR"):
            return "AR " + self.fake.word().title()
        if cname.endswith("_EN"):
            return self.fake.word().title()
        if "PIPELINE" in cname:
            return random.choice(["npc_pipeline", "gold_pipeline", "metrics_pipeline"])
        if cname == "ADF_PERIOD":
            return f"{self.pick_year()}-{int(np.random.randint(1,13)):02d}"
        if "URL" in cname:
            return f"https://example.com/{self.fake.slug()}"
        if "ACADEMIC_YEAR" in cname:
            y = self.pick_year()
            return f"{y}/{y+1}"
        if "QUARTER" in cname:
            return random.choice(["Q1", "Q2", "Q3", "Q4"])
        if "ITEM" in cname:
            return random.choice(["Revenue", "Expenditure", "Capital", "Operating", "Subsidy"])
        return self.fake.word().title()

    def generate_fact_rows(self, table: str, count: int) -> Iterable[Tuple]:
        cols = self.tables[table].column_names
        ycol = table_year_column(cols)
        has_triplet = all(c in cols for c in ("COUNTRY_ID", "INDICATOR_ID")) and ycol is not None
        used_triplet = set()

        i = 1
        generated = 0
        while generated < count:
            ctx: Dict[str, object] = {}
            row = []
            for c in cols:
                is_key_like = c.endswith("_ID") and (c.startswith(("GLD_", "SLVR_", "BRNZ_")) or c in self.id_pools)
                if self.should_null(table, c, is_key_like):
                    row.append(None)
                    continue
                val = self.make_value(table, c, i, ctx)
                ctx[c.upper()] = val
                row.append(val)

            if has_triplet:
                yv = ctx.get(ycol) if ycol else None
                triplet = (ctx.get("COUNTRY_ID"), yv, ctx.get("INDICATOR_ID"))
                if triplet in used_triplet:
                    i += 1
                    continue
                used_triplet.add(triplet)

            generated += 1
            i += 1
            yield tuple(row)

    def rows_for_table(self, table: str, count: int) -> Iterable[Tuple]:
        if classify_table_role(table) == "dimension":
            for row in self.generate_dimension_rows(table, count):
                yield row
        else:
            yield from self.generate_fact_rows(table, count)


def insert_batch(cur, table: str, cols: Sequence[str], rows: List[Tuple]) -> None:
    if not rows:
        return
    placeholders = ", ".join(["%s"] * len(cols))
    sql = f"INSERT INTO {qident(table)} ({', '.join(qident(c) for c in cols)}) VALUES ({placeholders})"
    cur.executemany(sql, rows)


def write_csv_rows(path: Path, rows: Iterable[Tuple]) -> int:
    import csv as _csv

    n = 0
    with path.open("w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        for row in rows:
            w.writerow(row)
            n += 1
    return n


def load_data_infile(cur, table: str, cols: Sequence[str], csv_file: Path) -> None:
    col_sql = ", ".join(qident(c) for c in cols)
    infile = str(csv_file).replace("\\", "\\\\")
    sql = (
        f"LOAD DATA LOCAL INFILE '{infile}' "
        f"INTO TABLE {qident(table)} "
        "FIELDS TERMINATED BY ',' ENCLOSED BY '\"' "
        "LINES TERMINATED BY '\\n' "
        f"({col_sql})"
    )
    cur.execute(sql)


def validate_counts(cur, expected: Dict[str, int]) -> Dict[str, int]:
    actual = {}
    for t in expected:
        cur.execute(f"SELECT COUNT(*) FROM {qident(t)}")
        actual[t] = int(cur.fetchone()[0])
    return actual


def validate_fk_like(cur, fk_edges: Dict[str, List[Tuple[str, str, str]]]) -> List[str]:
    issues = []
    for child, refs in fk_edges.items():
        for ccol, parent, pcol in refs:
            sql = (
                f"SELECT COUNT(*) FROM {qident(child)} c "
                f"LEFT JOIN {qident(parent)} p ON c.{qident(ccol)} = p.{qident(pcol)} "
                f"WHERE c.{qident(ccol)} IS NOT NULL AND p.{qident(pcol)} IS NULL"
            )
            cur.execute(sql)
            bad = int(cur.fetchone()[0])
            if bad > 0:
                issues.append(f"{child}.{ccol} -> {parent}.{pcol}: {bad} orphan rows")
    return issues


def validate_bounds(cur, tables: Dict[str, TableMeta]) -> List[str]:
    issues = []
    for tname, tmeta in tables.items():
        for c in tmeta.column_names:
            cname = c.upper()
            dtype = tmeta.dtype(c)
            if dtype not in ("DOUBLE", "FLOAT", "DECIMAL", "INT", "LONG"):
                continue
            if "PERCENT" in cname or "RATIO" in cname or "SCORE" in cname or ("INDEX" in cname and "RANK" not in cname):
                sql = f"SELECT COUNT(*) FROM {qident(tname)} WHERE {qident(c)} IS NOT NULL AND ({qident(c)} < 0 OR {qident(c)} > 100)"
                cur.execute(sql)
                bad = int(cur.fetchone()[0])
                if bad > 0:
                    issues.append(f"{tname}.{c}: {bad} out-of-range [0,100]")
    return issues


def mysql_literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (dt.datetime,)):
        return "'" + value.strftime("%Y-%m-%d %H:%M:%S") + "'"
    if isinstance(value, (dt.date,)):
        return "'" + value.strftime("%Y-%m-%d") + "'"
    if isinstance(value, (numbers.Integral, np.integer)):
        return str(int(value))
    if isinstance(value, (numbers.Real, np.floating)):
        v = float(value)
        if not math.isfinite(v):
            return "NULL"
        return repr(v)
    s = str(value).replace("\\", "\\\\").replace("'", "''")
    return "'" + s + "'"


def write_insert_sql(fh, table: str, cols: Sequence[str], rows: List[Tuple]) -> None:
    if not rows:
        return
    values_sql = []
    for row in rows:
        values_sql.append("(" + ", ".join(mysql_literal(v) for v in row) + ")")
    sql = f"INSERT INTO {qident(table)} ({', '.join(qident(c) for c in cols)}) VALUES\n  " + ",\n  ".join(values_sql) + ";\n"
    fh.write(sql)


def prepare_schema_sql(sql_path: Path, db_name: str) -> str:
    sql_text = sql_path.read_text(encoding="utf-8")
    sql_text = re.sub(r"CREATE\s+DATABASE\s+IF\s+NOT\s+EXISTS\s+`?\w+`?", f"CREATE DATABASE IF NOT EXISTS `{db_name}`", sql_text, flags=re.I)
    sql_text = re.sub(r"USE\s+`?\w+`?", f"USE `{db_name}`", sql_text, flags=re.I)
    return sql_text


def generate_sql_file(
    out_path: Path,
    schema_sql_text: str,
    tables: Dict[str, TableMeta],
    load_order: Sequence[str],
    allocation: Dict[str, int],
    gen: SyntheticGenerator,
    batch_size: int,
) -> Dict[str, int]:
    inserted: Dict[str, int] = {}
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("-- Auto-generated synthetic dataset SQL\n")
        fh.write("-- MySQL 5.7 compatible\n")
        fh.write("SET NAMES utf8mb4;\n")
        fh.write("SET FOREIGN_KEY_CHECKS=0;\n")
        fh.write("SET UNIQUE_CHECKS=0;\n")
        fh.write("SET AUTOCOMMIT=0;\n\n")
        fh.write(schema_sql_text.strip() + "\n\n")

        for tname in load_order:
            cols = tables[tname].column_names
            target = allocation.get(tname, 0)
            start = time.time()
            logging.info("Generating SQL for table %s target_rows=%d", tname, target)
            total = 0
            batch: List[Tuple] = []
            iterator = gen.rows_for_table(tname, target)
            if tqdm is not None:
                iterator = tqdm(iterator, total=target, desc=f"sql:{tname}", unit="row")
            for row in iterator:
                batch.append(row)
                if len(batch) >= batch_size:
                    write_insert_sql(fh, tname, cols, batch)
                    total += len(batch)
                    batch.clear()
            if batch:
                write_insert_sql(fh, tname, cols, batch)
                total += len(batch)
            inserted[tname] = total
            elapsed = time.time() - start
            logging.info("Wrote SQL inserts for %s rows=%d in %.2fs", tname, total, elapsed)

        fh.write("COMMIT;\n")
        fh.write("SET FOREIGN_KEY_CHECKS=1;\n")
        fh.write("SET UNIQUE_CHECKS=1;\n")
        fh.write("SET AUTOCOMMIT=1;\n")
    return inserted


def main() -> None:
    args = build_args()
    configure_logging(args.log_level)

    t0 = time.time()
    schema_path = Path(args.schema_sql)
    schema_csv_path = Path(args.schema_csv)
    metrics_csv_path = Path(args.metrics_csv)

    raw_tables = read_schema_csv(schema_csv_path)
    schema_table_names = parse_schema_table_names(schema_path)
    name_map = map_csv_tables_to_schema(list(raw_tables.keys()), schema_table_names)
    tables = remap_table_metadata(raw_tables, name_map)
    missing_tables = sorted(set(raw_tables.keys()) - set(name_map.keys()))
    if missing_tables:
        logging.warning("Schema.csv tables skipped (not found/mapped in Schema.sql): %d", len(missing_tables))
        for t in missing_tables[:20]:
            logging.warning("Skipped table: %s", t)
    metric_map = read_metric_map(metrics_csv_path, list(tables.keys()))

    dim_keys = infer_dimension_keys(tables)
    fk_edges = infer_fk_like_edges(tables, dim_keys)
    load_order = topo_load_order(tables, fk_edges)

    gen = SyntheticGenerator(args, tables, metric_map)
    allocation = gen.allocate_rows()
    if args.output_sql:
        out_path = Path(args.output_sql)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        schema_sql_text = prepare_schema_sql(schema_path, args.mysql_db)
        inserted = generate_sql_file(out_path, schema_sql_text, tables, load_order, allocation, gen, args.batch_size)
        logging.info("SQL file written to: %s", out_path)
    else:
        conn = connect_mysql(args)
        cur = conn.cursor()

        logging.info("Executing schema: %s", schema_path)
        execute_schema(conn, schema_path, args.mysql_db)

        cur.execute(f"USE {qident(args.mysql_db)}")
        cur.execute("SET autocommit=0")
        cur.execute("SET unique_checks=0")
        cur.execute("SET foreign_key_checks=0")

        emit_dir = Path(args.emit_csv) if args.emit_csv else None
        if emit_dir:
            emit_dir.mkdir(parents=True, exist_ok=True)

        inserted: Dict[str, int] = {}

        for tname in load_order:
            cols = tables[tname].column_names
            target = allocation.get(tname, 0)
            start = time.time()
            logging.info("Generating table %s target_rows=%d", tname, target)

            if emit_dir and args.use_load_data:
                csv_file = emit_dir / f"{tname}.csv"
                row_iter = gen.rows_for_table(tname, target)
                n = write_csv_rows(csv_file, row_iter)
                load_data_infile(cur, tname, cols, csv_file)
                conn.commit()
                inserted[tname] = n
            else:
                batch: List[Tuple] = []
                total = 0
                iterator = gen.rows_for_table(tname, target)
                if tqdm is not None:
                    iterator = tqdm(iterator, total=target, desc=tname, unit="row")
                for row in iterator:
                    batch.append(row)
                    if len(batch) >= args.batch_size:
                        insert_batch(cur, tname, cols, batch)
                        total += len(batch)
                        batch.clear()
                if batch:
                    insert_batch(cur, tname, cols, batch)
                    total += len(batch)
                conn.commit()
                inserted[tname] = total

            elapsed = time.time() - start
            logging.info("Loaded %s rows=%d in %.2fs", tname, inserted[tname], elapsed)

            if classify_table_role(tname) == "dimension":
                key_col = detect_key_column(tables[tname])
                if key_col and key_col not in gen.id_pools:
                    cur.execute(f"SELECT {qident(key_col)} FROM {qident(tname)}")
                    gen.id_pools[key_col] = [r[0] for r in cur.fetchall()]

        cur.execute("SET foreign_key_checks=1")
        cur.execute("SET unique_checks=1")
        conn.commit()

        if not args.skip_validation:
            logging.info("Running validations...")
            actual = validate_counts(cur, inserted)
            count_mismatch = [f"{t}: expected={inserted[t]} actual={actual[t]}" for t in inserted if inserted[t] != actual[t]]
            fk_issues = validate_fk_like(cur, fk_edges)
            bound_issues = validate_bounds(cur, tables)

            if count_mismatch:
                logging.warning("Count mismatches: %d", len(count_mismatch))
                for x in count_mismatch[:20]:
                    logging.warning("%s", x)
            if fk_issues:
                logging.warning("FK-like integrity issues: %d", len(fk_issues))
                for x in fk_issues[:20]:
                    logging.warning("%s", x)
            if bound_issues:
                logging.warning("Bound issues: %d", len(bound_issues))
                for x in bound_issues[:20]:
                    logging.warning("%s", x)

        cur.close()
        conn.close()

    total_time = time.time() - t0
    logging.info("Generation completed in %.2fs", total_time)
    print("\n=== TABLE ROW COUNTS ===")
    for t in sorted(inserted):
        print(f"{t}: {inserted[t]}")

if __name__ == "__main__":
    main()

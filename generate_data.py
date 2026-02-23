
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
    p.add_argument("--strict-quality", action="store_true", default=True, help="Enable strict quality validation gates")
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

MONTH_NAME_AR = {
    1: "يناير",
    2: "فبراير",
    3: "مارس",
    4: "أبريل",
    5: "مايو",
    6: "يونيو",
    7: "يوليو",
    8: "أغسطس",
    9: "سبتمبر",
    10: "أكتوبر",
    11: "نوفمبر",
    12: "ديسمبر",
}

EN_AR_EXACT_MAP = {
    "Export": "تصدير",
    "Import": "استيراد",
    "General": "عام",
    "Services": "خدمات",
    "Industry": "صناعة",
    "Trade": "تجارة",
    "Public": "عام",
    "Private": "خاص",
    "Male": "ذكر",
    "Female": "أنثى",
    "Single": "أعزب",
    "Married": "متزوج",
    "Divorced": "مطلق",
    "Outpatient": "عيادات خارجية",
    "Emergency": "طوارئ",
    "Inpatient": "تنويم",
    "Revenue": "إيرادات",
    "Expenditure": "مصروفات",
    "Capital": "رأسمالي",
    "Operating": "تشغيلي",
    "Subsidy": "دعم",
    "Up": "ارتفاع",
    "Down": "انخفاض",
    "Stable": "مستقر",
    "Qatar": "قطر",
    "Saudi Arabia": "المملكة العربية السعودية",
    "United Arab Emirates": "الإمارات العربية المتحدة",
    "United States": "الولايات المتحدة",
    "Kuwait": "الكويت",
    "Oman": "عمان",
    "Bahrain": "البحرين",
    "United Kingdom": "المملكة المتحدة",
    "Germany": "ألمانيا",
    "France": "فرنسا",
    "China": "الصين",
    "Japan": "اليابان",
    "India": "الهند",
    "South Korea": "كوريا الجنوبية",
    "Australia": "أستراليا",
    "Canada": "كندا",
    "Spain": "إسبانيا",
    "Italy": "إيطاليا",
    "Netherlands": "هولندا",
    "Sweden": "السويد",
    "Qatari": "قطري",
    "Saudi": "سعودي",
    "Emirati": "إماراتي",
    "American": "أمريكي",
    "German": "ألماني",
    "Kuwaiti": "كويتي",
    "Omani": "عماني",
    "Bahraini": "بحريني",
    "British": "بريطاني",
    "French": "فرنسي",
    "Chinese": "صيني",
    "Japanese": "ياباني",
    "Indian": "هندي",
    "Korean": "كوري",
    "Australian": "أسترالي",
    "Canadian": "كندي",
    "Spanish": "إسباني",
    "Italian": "إيطالي",
    "Dutch": "هولندي",
    "Swedish": "سويدي",
    "Employed": "موظف",
    "Unemployed": "عاطل عن العمل",
    "Not in labor force": "غير نشط اقتصاديا",
    "Residential": "سكني",
    "Commercial": "تجاري",
    "Industrial": "صناعي",
    "Government": "حكومي",
    "Qatari Riyal": "ريال قطري",
    "UAE Dirham": "درهم إماراتي",
    "Saudi Riyal": "ريال سعودي",
    "Kuwaiti Dinar": "دينار كويتي",
    "Omani Rial": "ريال عماني",
    "Bahraini Dinar": "دينار بحريني",
    "US Dollar": "دولار أمريكي",
    "Pound Sterling": "جنيه إسترليني",
    "Euro": "يورو",
    "Yuan": "يوان",
    "Yen": "ين",
    "Indian Rupee": "روبية هندية",
    "Won": "وون",
    "Australian Dollar": "دولار أسترالي",
    "Canadian Dollar": "دولار كندي",
    "Swedish Krona": "كرونة سويدية",
    "Middle East": "الشرق الأوسط",
    "North America": "أمريكا الشمالية",
    "Europe": "أوروبا",
    "Asia": "آسيا",
    "Oceania": "أوقيانوسيا",
    "Developed": "متقدم",
    "Emerging": "ناشئ",
    "Frontier": "حدودي",
    "High income": "دخل مرتفع",
    "Upper middle income": "دخل متوسط مرتفع",
    "Lower middle income": "دخل متوسط منخفض",
    "Low income": "دخل منخفض",
    "Managers": "مديرون",
    "Professionals": "مهنيون",
    "Technicians": "فنيون",
    "Service Workers": "عاملو خدمات",
    "Clerical": "وظائف مكتبية",
    "Tourism": "سياحة",
    "Manufacturing": "تصنيع",
    "Education": "تعليم",
    "Healthcare and Social Services": "الصحة والخدمات الاجتماعية",
    "Information and Communication": "المعلومات والاتصالات",
    "Finance and Insurance": "التمويل والتأمين",
    "Transport and Storage": "النقل والتخزين",
    "Agriculture and Livestock": "الزراعة والثروة الحيوانية",
    "Academic Services": "الخدمات الأكاديمية",
    "Administrative Services": "الخدمات الإدارية",
    "Grade 1": "الصف 1",
    "Grade 2": "الصف 2",
    "Grade 3": "الصف 3",
    "Grade 4": "الصف 4",
    "Grade 5": "الصف 5",
    "Grade 6": "الصف 6",
    "Grade 7": "الصف 7",
    "Grade 8": "الصف 8",
    "Grade 9": "الصف 9",
    "Grade 10": "الصف 10",
    "Grade 11": "الصف 11",
    "Grade 12": "الصف 12",
    "Academic": "أكاديمي",
    "Technical": "تقني",
    "Vocational": "مهني",
    "University": "جامعي",
    "Foundational": "تأسيسي",
    "Applied": "تطبيقي",
    "Advanced": "متقدم",
    "Food Commodities": "السلع الغذائية",
    "Industrial Materials": "المواد الصناعية",
    "Medical Supplies": "المستلزمات الطبية",
    "Tax Revenue": "الإيرادات الضريبية",
    "Non-tax Revenue": "الإيرادات غير الضريبية",
    "Investment Income": "دخل الاستثمارات",
    "Grants": "المنح",
    "Compensation of Employees": "تعويضات الموظفين",
    "Goods and Services": "السلع والخدمات",
    "Social Benefits": "المنافع الاجتماعية",
    "Subsidies": "الدعم",
    "Infrastructure Projects": "مشاريع البنية التحتية",
    "Capital Transfers": "التحويلات الرأسمالية",
    "Asset Acquisition": "اقتناء الأصول",
    "Development Projects": "مشاريع التنمية",
    "Utilities": "المرافق",
    "Operations and Maintenance": "التشغيل والصيانة",
    "Administrative Costs": "التكاليف الإدارية",
    "Service Contracts": "عقود الخدمات",
    "Treasury Bills": "أذونات الخزانة",
    "Treasury Bonds": "سندات الخزانة",
    "Sukuk": "صكوك",
    "Repo Operations": "عمليات إعادة الشراء",
    "Additional Economic Indicators": "المؤشرات الاقتصادية الإضافية",
    "Economic Diversification": "التنويع الاقتصادي",
    "Administrative Data": "بيانات إدارية",
    "Survey Data": "بيانات مسحية",
    "International Database": "قاعدة بيانات دولية",
    "World Bank": "البنك الدولي",
    "International Monetary Fund": "صندوق النقد الدولي",
    "Ministry of Planning": "وزارة التخطيط",
    "National Statistics Authority": "جهاز الإحصاء الوطني",
    "Economic Pillar": "ركيزة اقتصادية",
    "Social Pillar": "ركيزة اجتماعية",
    "Environmental Pillar": "ركيزة بيئية",
    "Institutional Pillar": "ركيزة مؤسسية",
    "Productivity Outcome": "مخرجات الإنتاجية",
    "Inclusion Outcome": "مخرجات الشمول",
    "Sustainability Outcome": "مخرجات الاستدامة",
    "Human Capital": "رأس المال البشري",
    "Sustainable Development": "التنمية المستدامة",
    "Percent": "نسبة مئوية",
    "Index": "مؤشر",
    "Ranking": "ترتيب",
    "QAR Million": "مليون ريال قطري",
    "Hours": "ساعات",
}

EN_AR_WORD_MAP = {
    "export": "تصدير",
    "import": "استيراد",
    "general": "عام",
    "services": "خدمات",
    "industry": "صناعة",
    "trade": "تجارة",
    "public": "عام",
    "private": "خاص",
    "male": "ذكر",
    "female": "أنثى",
    "single": "أعزب",
    "married": "متزوج",
    "divorced": "مطلق",
    "outpatient": "عيادات خارجية",
    "emergency": "طوارئ",
    "inpatient": "تنويم",
    "revenue": "إيرادات",
    "expenditure": "مصروفات",
    "capital": "رأسمالي",
    "operating": "تشغيلي",
    "subsidy": "دعم",
    "up": "ارتفاع",
    "down": "انخفاض",
    "stable": "مستقر",
    "qatar": "قطر",
    "saudi": "سعودي",
    "arabia": "العربية",
    "united": "المتحدة",
    "arab": "عربي",
    "emirates": "الإمارات",
    "states": "الولايات",
    "gulf": "الخليج",
    "middle": "الوسطى",
    "east": "الشرق",
    "north": "الشمال",
    "america": "أمريكا",
    "europe": "أوروبا",
    "asia": "آسيا",
    "oceania": "أوقيانوسيا",
    "qatari": "قطري",
    "emirati": "إماراتي",
    "kuwaiti": "كويتي",
    "omani": "عماني",
    "bahraini": "بحريني",
    "british": "بريطاني",
    "french": "فرنسي",
    "chinese": "صيني",
    "japanese": "ياباني",
    "indian": "هندي",
    "korean": "كوري",
    "australian": "أسترالي",
    "canadian": "كندي",
    "spanish": "إسباني",
    "italian": "إيطالي",
    "dutch": "هولندي",
    "swedish": "سويدي",
    "employed": "موظف",
    "unemployed": "عاطل",
    "residential": "سكني",
    "commercial": "تجاري",
    "industrial": "صناعي",
    "government": "حكومي",
    "health": "الصحة",
    "economic": "اقتصادي",
    "diversification": "تنويع",
    "human": "بشري",
    "indicator": "مؤشر",
    "value": "قيمة",
    "target": "مستهدف",
    "latest": "أحدث",
    "measurement": "قياس",
    "unit": "وحدة",
    "source": "مصدر",
    "frequency": "تكرار",
    "annual": "سنوي",
    "quarterly": "ربع سنوي",
    "monthly": "شهري",
    "weekly": "أسبوعي",
    "daily": "يومي",
    "status": "حالة",
    "usage": "استخدام",
    "sector": "قطاع",
    "level": "مستوى",
    "track": "مسار",
    "skill": "مهارة",
    "group": "مجموعة",
    "type": "نوع",
    "category": "فئة",
    "band": "فئة",
    "item": "عنصر",
    "occupation": "مهنة",
    "tier": "فئة",
    "rank": "ترتيب",
    "administrative": "إداري",
    "data": "بيانات",
    "database": "قاعدة بيانات",
    "survey": "مسح",
    "national": "وطني",
    "development": "تنمية",
    "strategy": "استراتيجية",
    "cluster": "عنقود",
    "pillar": "ركيزة",
    "outcome": "ناتج",
    "mapping": "مواءمة",
    "description": "وصف",
    "quality": "جودة",
    "score": "درجة",
    "high": "مرتفع",
    "medium": "متوسط",
    "low": "منخفض",
    "gdp": "الناتج المحلي الإجمالي",
    "nds": "استراتيجية التنمية الوطنية",
}

ARABIC_FILLER_VALUES = {"عام", "فئة", "نوع", "قيمة"}
PLACEHOLDER_RE = re.compile(r"^(Status \d+|Usage \d+|Type [A-Z]|Category [A-Z]|Item \d+|Indicator \d+|Academic Stage \d+)$")

CURATED_DOMAINS: Dict[str, List[str]] = {
    "STATUS_IN_EMPLOYMENT": ["Employed", "Unemployed", "Not in labor force"],
    "PROPERTY_USAGE": ["Residential", "Commercial", "Industrial", "Government"],
    "VISITORS_SUBCATEGORY": ["Private", "Trade", "Services", "Industry", "Tourism"],
    "TRADE_TYPE": ["Import", "Export"],
    "INCOME_GROUP": ["High income", "Upper middle income", "Lower middle income", "Low income"],
    "MARKET_TYPE": ["Developed", "Emerging", "Frontier"],
    "PROPERTY_TYPE": ["Residential", "Commercial", "Industrial", "Mixed Use"],
    "OCCUPATION": ["Managers", "Professionals", "Technicians", "Service Workers", "Clerical"],
    "RATING": ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-"],
    "TIER_PERFORMANCE": ["Tier 1", "Tier 2", "Tier 3", "Tier 4"],
    "TIER_RANK": ["Top quartile", "Upper middle", "Lower middle", "Bottom quartile"],
    "ENTITY": ["Government Entity", "Private Entity", "Public Institution", "International Organization"],
    "BUDGET_CATEGORY": ["Revenue", "Expenditure", "Capital", "Operating"],
    "TREASURY_ITEM": ["Treasury Bills", "Treasury Bonds", "Sukuk", "Repo Operations"],
    "ACADEMIC_STAGE": [f"Grade {i}" for i in range(1, 13)],
    "ACTIVITY_TYPE": ["Manufacturing", "Trade", "Services", "Tourism", "Logistics"],
    "FIRM_STAGE": ["Startup", "Growth", "Mature", "Expansion"],
}

BUDGET_ITEMS_BY_CATEGORY: Dict[str, List[str]] = {
    "Revenue": ["Tax Revenue", "Non-tax Revenue", "Investment Income", "Grants"],
    "Expenditure": ["Compensation of Employees", "Goods and Services", "Social Benefits", "Subsidies"],
    "Capital": ["Infrastructure Projects", "Capital Transfers", "Asset Acquisition", "Development Projects"],
    "Operating": ["Utilities", "Operations and Maintenance", "Administrative Costs", "Service Contracts"],
}

SECTOR_ACTIVITY_AR = {
    "Economic Diversification": "التنويع الاقتصادي",
    "Healthcare and Social Services": "الصحة والخدمات الاجتماعية",
    "Information and Communication": "المعلومات والاتصالات",
    "Transport and Storage": "النقل والتخزين",
    "Finance and Insurance": "التمويل والتأمين",
    "Education": "التعليم",
    "Manufacturing": "التصنيع",
    "Agriculture and Livestock": "الزراعة والثروة الحيوانية",
}

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
        self.gender_by_id: Dict[int, Tuple[str, str]] = {1: ("Male", "ذكر"), 2: ("Female", "أنثى")}
        self.indicator_trend_by_id: Dict[int, str] = {}
        self.status_by_id: Dict[int, Tuple[str, str]] = {}
        self.property_usage_by_id: Dict[int, Tuple[str, str]] = {}
        self.marital_by_id: Dict[int, Tuple[str, str]] = {}
        self.visit_type_by_id: Dict[int, Tuple[str, str]] = {}
        self.country_by_id: Dict[int, Dict[str, str]] = {}
        self.establishment_sector_by_id: Dict[int, Tuple[str, str]] = {}
        self.indicator_name_by_id: Dict[int, str] = {}
        self.trade_type_by_id: Dict[int, Tuple[str, str]] = {
            1: ("Import", "استيراد"),
            2: ("Export", "تصدير"),
        }
        self.translation_safe_choices: Dict[str, List[str]] = {
            "COUNTRY_NAME": [x[2] for x in COUNTRY_DATA],
            "NATIONALITY": [x[3] for x in COUNTRY_DATA],
            "CURRENCY": [x[5] for x in COUNTRY_DATA],
            "REGION": sorted({x[6] for x in COUNTRY_DATA}),
        }

        self.null_rates: Dict[Tuple[str, str], float] = {}
        for tname, tmeta in tables.items():
            for c in tmeta.column_names:
                self.null_rates[(tname, c)] = random.uniform(args.null_rate_min, args.null_rate_max)
        self.table_pair_cache: Dict[str, List[Tuple[str, str]]] = {}
        self.translation_protected_cols: Dict[str, set] = {}
        for tname, tmeta in tables.items():
            cols = set(tmeta.column_names)
            protected = set()
            for c in tmeta.column_names:
                if c.endswith("_AR"):
                    protected.add(c)
                    en = c[:-3] + "_EN"
                    base = c[:-3]
                    if en in cols:
                        protected.add(en)
                    if base in cols:
                        protected.add(base)
                if c.endswith("_EN") and (c[:-3] + "_AR" in cols):
                    protected.add(c)
                if (c + "_AR") in cols:
                    protected.add(c)
            self.translation_protected_cols[tname] = protected
        self.column_domains = self.build_column_domains_from_sources()

    @staticmethod
    def split_option_values(value: str) -> List[str]:
        out: List[str] = []
        for token in re.split(r"[\n,|]+", value):
            t = token.strip()
            if not t:
                continue
            if re.fullmatch(r"[A-Z0-9_]+", t):
                continue
            if len(t) > 80:
                continue
            out.append(t)
        return out

    def build_column_domains_from_sources(self) -> Dict[str, List[str]]:
        out: Dict[str, set] = defaultdict(set)
        for _tbl, rows in self.metric_map.items():
            for row in rows:
                if not isinstance(row, dict):
                    continue
                for key, val in row.items():
                    if val is None:
                        continue
                    txt = str(val).strip()
                    if not txt:
                        continue
                    k = str(key).strip().lower()
                    if "sector" in k and "\n" not in txt and "," not in txt:
                        out["SECTOR"].add(txt)
                    if "source" in k and "entity" in k and "\n" not in txt and "," not in txt:
                        out["SOURCE_ENTITY"].add(txt)
                    if "frequency" in k and "\n" not in txt and "," not in txt:
                        out["FREQUENCY"].add(txt)
                    if "type" in k and "\n" not in txt and "," not in txt:
                        out["TYPE"].add(txt)
                    if "group" in k and "\n" not in txt and "," not in txt:
                        out["GROUP"].add(txt)
                    if "category" in k and "\n" not in txt and "," not in txt:
                        out["CATEGORY"].add(txt)
                    if "market" in k and "type" in k and "\n" not in txt and "," not in txt:
                        out["MARKET_TYPE"].add(txt)
                    if "income" in k and "group" in k and "\n" not in txt and "," not in txt:
                        out["INCOME_GROUP"].add(txt)
                    if "property" in k and "type" in k and "\n" not in txt and "," not in txt:
                        out["PROPERTY_TYPE"].add(txt)
                    if "property" in k and "usage" in k and "\n" not in txt and "," not in txt:
                        out["PROPERTY_USAGE"].add(txt)
                    if "occupation" in k and "\n" not in txt and "," not in txt:
                        out["OCCUPATION"].add(txt)
                    if "x-axis options" in k or "group by options" in k or "filter options" in k:
                        for opt in self.split_option_values(txt):
                            lo = opt.lower()
                            if "trade_flow" in lo:
                                out["TRADE_TYPE"].update(["Import", "Export"])
                            if "visitors_subcategory" in lo:
                                out["VISITORS_SUBCATEGORY"].update(CURATED_DOMAINS["VISITORS_SUBCATEGORY"])
                            if lo in ("category", "item"):
                                out["BUDGET_CATEGORY"].update(CURATED_DOMAINS["BUDGET_CATEGORY"])
                                for items in BUDGET_ITEMS_BY_CATEGORY.values():
                                    out["ITEM"].update(items)
                            if "academic_stage" in lo:
                                out["ACADEMIC_STAGE"].update(CURATED_DOMAINS["ACADEMIC_STAGE"])
                            if "activity_type" in lo:
                                out["ACTIVITY_TYPE"].update(CURATED_DOMAINS["ACTIVITY_TYPE"])
                            if "firm_stage" in lo:
                                out["FIRM_STAGE"].update(CURATED_DOMAINS["FIRM_STAGE"])

        # Strict constrained defaults when source rows do not provide enumerations.
        for key, vals in CURATED_DOMAINS.items():
            if not out[key]:
                out[key].update(vals)
        if not out["ITEM"]:
            for items in BUDGET_ITEMS_BY_CATEGORY.values():
                out["ITEM"].update(items)
        if not out["BUDGET_CATEGORY"]:
            out["BUDGET_CATEGORY"].update(CURATED_DOMAINS["BUDGET_CATEGORY"])
        if not out["SECTOR"] and out["CATEGORY"]:
            out["SECTOR"].update(out["CATEGORY"])
        return {k: sorted(v) for k, v in out.items()}

    @staticmethod
    def metric_get(row: dict, *keys: str) -> Optional[str]:
        if not isinstance(row, dict):
            return None
        norm = {str(k).strip().lower(): v for k, v in row.items()}
        for key in keys:
            v = norm.get(str(key).strip().lower())
            if v is None:
                continue
            s = str(v).strip()
            if s:
                return s
        return None

    def translate_en_to_ar(self, value: object) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if not text:
            return ""
        if text in EN_AR_EXACT_MAP:
            return EN_AR_EXACT_MAP[text]

        tokens = re.split(r"(\W+)", text)
        out: List[str] = []
        translit = {
            "a": "ا", "b": "ب", "c": "ك", "d": "د", "e": "ي", "f": "ف", "g": "ج", "h": "ه",
            "i": "ي", "j": "ج", "k": "ك", "l": "ل", "m": "م", "n": "ن", "o": "و", "p": "ب",
            "q": "ق", "r": "ر", "s": "س", "t": "ت", "u": "و", "v": "ف", "w": "و", "x": "كس",
            "y": "ي", "z": "ز",
        }
        for tok in tokens:
            if not tok:
                continue
            key = tok.lower()
            if key in EN_AR_WORD_MAP:
                out.append(EN_AR_WORD_MAP[key])
            elif re.fullmatch(r"[A-Za-z_]+", tok):
                out.append("".join(translit.get(ch, "") for ch in key if ch.isalpha()) or "قيمة")
            else:
                out.append(tok)
        translated = "".join(out).strip()
        if re.fullmatch(r"[\d\.\,\-\+\%\s/]+", translated):
            return translated
        if not translated or translated in ARABIC_FILLER_VALUES:
            return "عام"
        if re.search(r"[A-Za-z]", translated) or not re.search(r"[\u0600-\u06FF]", translated):
            return "عام"
        return translated if translated else "عام"

    def default_english_for_pair(self, col_name: str) -> str:
        c = col_name.upper()
        if "COUNTRY_NAME" in c:
            return random.choice(self.translation_safe_choices["COUNTRY_NAME"])
        if "NATIONALITY" in c:
            return random.choice(self.translation_safe_choices["NATIONALITY"])
        if "CURRENCY" in c:
            return random.choice(self.translation_safe_choices["CURRENCY"])
        if "REGION" in c:
            return random.choice(self.translation_safe_choices["REGION"])
        if "GENDER" in c:
            return random.choice(["Male", "Female"])
        if "MARITAL_STATUS" in c:
            return random.choice(["Single", "Married", "Divorced"])
        if "VISIT_TYPE" in c:
            return random.choice(["Outpatient", "Emergency", "Inpatient"])
        if "TREND_DIRECTION" in c:
            return random.choice(["Up", "Down", "Stable"])
        if "INCOME_GROUP" in c:
            return random.choice(self.column_domains.get("INCOME_GROUP", CURATED_DOMAINS["INCOME_GROUP"]))
        if "MARKET_TYPE" in c:
            return random.choice(self.column_domains.get("MARKET_TYPE", CURATED_DOMAINS["MARKET_TYPE"]))
        if "PROPERTY_TYPE" in c:
            return random.choice(self.column_domains.get("PROPERTY_TYPE", CURATED_DOMAINS["PROPERTY_TYPE"]))
        if "PROPERTY_USAGE" in c:
            return random.choice(self.column_domains.get("PROPERTY_USAGE", CURATED_DOMAINS["PROPERTY_USAGE"]))
        if "STATUS_IN_EMPLOYMENT" in c:
            return random.choice(self.column_domains.get("STATUS_IN_EMPLOYMENT", CURATED_DOMAINS["STATUS_IN_EMPLOYMENT"]))
        if "OCCUPATION" in c:
            return random.choice(self.column_domains.get("OCCUPATION", CURATED_DOMAINS["OCCUPATION"]))
        if "SECTOR" in c:
            return random.choice(self.column_domains.get("SECTOR", CURATED_DOMAINS["ACTIVITY_TYPE"]))
        if "CATEGORY" in c:
            return random.choice(self.column_domains.get("BUDGET_CATEGORY", CURATED_DOMAINS["BUDGET_CATEGORY"]))
        if "GROUP" in c:
            return random.choice(self.column_domains.get("GROUP", ["Group A", "Group B"]))
        if "TYPE" in c:
            if "VISITORS_SUBCATEGORY" in c:
                return random.choice(self.column_domains.get("VISITORS_SUBCATEGORY", CURATED_DOMAINS["VISITORS_SUBCATEGORY"]))
            if "TRADE_TYPE" in c:
                return random.choice(self.column_domains.get("TRADE_TYPE", CURATED_DOMAINS["TRADE_TYPE"]))
            return random.choice(self.column_domains.get("TYPE", ["Primary", "Secondary"]))
        if "ITEM" in c:
            return random.choice(self.column_domains.get("ITEM", CURATED_DOMAINS["TREASURY_ITEM"]))
        if c == "ENTITY_EN":
            return random.choice(self.column_domains.get("ENTITY", CURATED_DOMAINS["ENTITY"]))
        if c == "ACADEMIC_STAGE":
            return random.choice(self.column_domains.get("ACADEMIC_STAGE", CURATED_DOMAINS["ACADEMIC_STAGE"]))
        if "FREQUENCY" in c:
            return random.choice(self.column_domains.get("FREQUENCY", ["Annual", "Quarterly", "Monthly"]))
        if "SOURCE" in c:
            return random.choice(self.column_domains.get("SOURCE_ENTITY", ["Source A", "Source B"]))
        if "DESCRIPTION" in c or "DEFINITION" in c:
            return "Descriptive metadata"
        return "Standard"

    @staticmethod
    def is_controlled_column(col_name: str) -> bool:
        c = col_name.upper()
        if c.endswith("_EN") or c.endswith("_AR"):
            return True
        if any(k in c for k in ["STATUS", "USAGE", "OCCUPATION", "INCOME_GROUP", "MARKET_TYPE", "PROPERTY_TYPE", "RATING", "TIER", "REGION", "CATEGORY", "TYPE", "GROUP", "SECTOR", "ITEM", "NATIONALITY", "GENDER", "DESCRIPTION", "DEFINITION", "SOURCE", "ACADEMIC_STAGE", "INDICATOR"]):
            return True
        if any(k in c for k in ["RATIO", "SCORE", "RANK", "GDP_PER_CAPITA", "POPULATION", "VALUE"]) and not c.endswith("_ID"):
            return True
        return False

    def controlled_text_value(self, table: str, col_name: str, row_ctx: Optional[Dict[str, object]] = None) -> Optional[str]:
        c = col_name.upper()
        ctx = row_ctx or {}
        if table == "mof_fact_national_budget" and c == "CATEGORY_EN":
            return random.choice(self.column_domains.get("BUDGET_CATEGORY", CURATED_DOMAINS["BUDGET_CATEGORY"]))
        if table == "mof_fact_national_budget" and c == "CATEGORY_AR":
            en = ctx.get("CATEGORY_EN") or random.choice(self.column_domains.get("BUDGET_CATEGORY", CURATED_DOMAINS["BUDGET_CATEGORY"]))
            return self.translate_en_to_ar(en)
        if table == "mof_fact_national_budget" and c == "ITEM_EN":
            cat = ctx.get("CATEGORY_EN")
            if cat in BUDGET_ITEMS_BY_CATEGORY:
                return random.choice(BUDGET_ITEMS_BY_CATEGORY[str(cat)])
            return random.choice(self.column_domains.get("ITEM", CURATED_DOMAINS["TREASURY_ITEM"]))
        if table == "mof_fact_national_budget" and c == "ITEM_AR":
            en = ctx.get("ITEM_EN")
            if not en:
                cat = ctx.get("CATEGORY_EN")
                if cat in BUDGET_ITEMS_BY_CATEGORY:
                    en = random.choice(BUDGET_ITEMS_BY_CATEGORY[str(cat)])
                else:
                    en = random.choice(self.column_domains.get("ITEM", CURATED_DOMAINS["TREASURY_ITEM"]))
            return self.translate_en_to_ar(en)
        if table == "mof_fact_return_on_treasury" and c == "ITEM_EN":
            return random.choice(self.column_domains.get("TREASURY_ITEM", CURATED_DOMAINS["TREASURY_ITEM"]))
        if table == "mof_fact_return_on_treasury" and c == "ITEM_AR":
            en = ctx.get("ITEM_EN") or random.choice(self.column_domains.get("TREASURY_ITEM", CURATED_DOMAINS["TREASURY_ITEM"]))
            return self.translate_en_to_ar(en)
        if c == "STATUS_IN_EMPLOYMENT_EN":
            return random.choice(self.column_domains["STATUS_IN_EMPLOYMENT"])
        if c == "PROPERTY_USAGE_EN":
            return random.choice(self.column_domains["PROPERTY_USAGE"])
        if c == "MAIN_OCCUPATION":
            return random.choice(self.column_domains["OCCUPATION"])
        if c == "MARKET_TYPE":
            return random.choice(self.column_domains["MARKET_TYPE"])
        if c == "INCOME_GROUP":
            return random.choice(self.column_domains["INCOME_GROUP"])
        if c == "PROPERTY_TYPE":
            return random.choice(self.column_domains["PROPERTY_TYPE"])
        if c == "VISITORS_SUBCATEGORY_EN":
            return random.choice(self.column_domains.get("VISITORS_SUBCATEGORY", CURATED_DOMAINS["VISITORS_SUBCATEGORY"]))
        if c == "TRADE_TYPE_EN":
            return random.choice(self.column_domains.get("TRADE_TYPE", CURATED_DOMAINS["TRADE_TYPE"]))
        if c == "ACADEMIC_STAGE":
            return random.choice(self.column_domains.get("ACADEMIC_STAGE", CURATED_DOMAINS["ACADEMIC_STAGE"]))
        if c == "ENTITY_EN":
            return random.choice(self.column_domains.get("ENTITY", CURATED_DOMAINS["ENTITY"]))
        if c in ("ITEM", "ITEM_EN") and table not in ("mof_fact_national_budget", "mof_fact_return_on_treasury"):
            return random.choice(self.column_domains.get("ITEM", CURATED_DOMAINS["TREASURY_ITEM"]))
        if "RATING" in c and not c.endswith("_AR"):
            return random.choice(self.column_domains["RATING"])
        if c == "TIER_PERFORMANCE":
            return random.choice(self.column_domains["TIER_PERFORMANCE"])
        if c == "TIER_RANK":
            return random.choice(self.column_domains["TIER_RANK"])
        if c == "QUARTER_LABEL":
            return random.choice(["Q1", "Q2", "Q3", "Q4"])
        if c in ("GENDER", "GENDER_EN"):
            return random.choice(["Male", "Female"])
        if c == "NATIONALITY":
            return random.choice(self.translation_safe_choices["NATIONALITY"])
        if c == "REGION":
            return random.choice(self.translation_safe_choices["REGION"])
        if c.startswith("LPI_") and c.endswith("GROUPED_RANK"):
            return random.choice(self.column_domains.get("TIER_RANK", ["Rank 1", "Rank 2", "Rank 3", "Rank 4"]))
        if c in ("KPI_VALUE", "KPI_TIME_PERIOD"):
            return random.choice(["Annual", "Quarterly", "Monthly", "Stable", "Improving", "Declining"])
        if "DESCRIPTION" in c or "DEFINITION" in c:
            return "General description"
        if "SOURCE" in c:
            return random.choice(self.column_domains.get("SOURCE_ENTITY", ["Source A", "Source B"]))
        if table == "intl_fact_out_of_pocket_expenditure" and c == "INDICATOR":
            iid = self.id_pools.get("INDICATOR_ID", [])
            if iid:
                picked = random.choice(iid)
                return self.indicator_name_by_id.get(int(picked), "Out-of-pocket Expenditure")
            return "Out-of-pocket Expenditure"
        if table == "moci_fact_manufacturing_sector_total_investments" and c == "FIRM_NAME_EN":
            return random.choice(["Qatar Industrial Co.", "Doha Manufacturing Group", "Gulf Precision Works", "National Processing LLC"])
        if table == "moci_fact_manufacturing_sector_total_investments" and c == "ACTIVITY_TYPE":
            return random.choice(self.column_domains.get("ACTIVITY_TYPE", CURATED_DOMAINS["ACTIVITY_TYPE"]))
        if table == "moci_fact_manufacturing_sector_total_investments" and c == "FIRM_STAGE":
            return random.choice(self.column_domains.get("FIRM_STAGE", CURATED_DOMAINS["FIRM_STAGE"]))
        if table == "moci_fact_strategic_commodities_reserve" and c == "GROUP_NAME_EN":
            return random.choice(["Food Commodities", "Industrial Materials", "Medical Supplies"])
        if table == "moci_fact_strategic_commodities_reserve" and c == "GROUP_NAME_AR":
            return self.translate_en_to_ar(random.choice(["Food Commodities", "Industrial Materials", "Medical Supplies"]))
        return None

    def ensure_arabic_text(self, value: object) -> str:
        s = "" if value is None else str(value).strip()
        if not s:
            return "قيمة"
        if re.search(r"[A-Za-z]", s):
            return self.translate_en_to_ar(s)
        return s

    def en_ar_pairs(self, table: str) -> List[Tuple[str, str]]:
        if table in self.table_pair_cache:
            return self.table_pair_cache[table]
        cols = set(self.tables[table].column_names)
        pairs: List[Tuple[str, str]] = []
        for c in self.tables[table].column_names:
            if c.endswith("_AR"):
                en = c[:-3] + "_EN"
                if en in cols:
                    pairs.append((en, c))
                    continue
                # Some schemas use BASE + _AR where the English/source column is BASE (no _EN suffix).
                base = c[:-3]
                if base in cols:
                    pairs.append((base, c))
        self.table_pair_cache[table] = pairs
        return pairs

    def enforce_ar_pairs_for_row(self, table: str, cols: Sequence[str], row_values: List[object], metric_hint: Optional[dict] = None) -> None:
        idx_map = {c: i for i, c in enumerate(cols)}

        # Table-specific hard rule from source-of-truth request
        if table == "gac_fact_time_to_import_and_export":
            en_idx = idx_map.get("CATEGORY_EN")
            ar_idx = idx_map.get("CATEGORY_AR")
            if en_idx is not None:
                en_val = row_values[en_idx]
                if en_val not in ("Export", "Import"):
                    en_val = random.choice(["Export", "Import"])
                    row_values[en_idx] = en_val
                if ar_idx is not None:
                    row_values[ar_idx] = EN_AR_EXACT_MAP.get(str(en_val), self.translate_en_to_ar(en_val))

        for en_col, ar_col in self.en_ar_pairs(table):
            en_idx = idx_map.get(en_col)
            ar_idx = idx_map.get(ar_col)
            if en_idx is None or ar_idx is None:
                continue
            en_val = row_values[en_idx]
            if en_val is None or str(en_val).strip() == "":
                en_val = self.default_english_for_pair(en_col)
                row_values[en_idx] = en_val

            if ar_col in ("INDICATOR_DEFINITION_AR", "QDTI_DESCRIPTION_AR"):
                existing_ar = row_values[ar_idx]
                if existing_ar is not None:
                    ar_s = str(existing_ar).strip()
                    if ar_s and re.search(r"[\u0600-\u06FF]", ar_s) and not re.search(r"[A-Za-z]", ar_s) and ar_s not in ARABIC_FILLER_VALUES:
                        continue

            if metric_hint and table == "dim_indicators":
                if ar_col == "INDICATOR_DEFINITION_AR":
                    ar_text = self.metric_get(metric_hint, "Short Description AR", "Short Description AR ")
                    row_values[ar_idx] = self.ensure_arabic_text(ar_text if ar_text else self.translate_en_to_ar(en_val))
                    continue

            row_values[ar_idx] = self.ensure_arabic_text(self.translate_en_to_ar(en_val))

    def enforce_id_consistency_for_row(self, table: str, cols: Sequence[str], row_values: List[object]) -> None:
        idx_map = {c: i for i, c in enumerate(cols)}

        # Keep gender text values consistent with GENDER_ID.
        gid_idx = idx_map.get("GENDER_ID")
        if gid_idx is not None and row_values[gid_idx] is not None:
            try:
                gid = int(row_values[gid_idx])
            except Exception:
                gid = None
            if gid in self.gender_by_id:
                g_en, g_ar = self.gender_by_id[gid]
                if "GENDER_EN" in idx_map:
                    row_values[idx_map["GENDER_EN"]] = g_en
                if "GENDER_AR" in idx_map:
                    row_values[idx_map["GENDER_AR"]] = g_ar

        # Keep trend direction consistent with INDICATOR_ID when both exist.
        iid_idx = idx_map.get("INDICATOR_ID")
        trend_idx = idx_map.get("TREND_DIRECTION")
        if iid_idx is not None and trend_idx is not None and row_values[iid_idx] is not None:
            try:
                iid = int(row_values[iid_idx])
            except Exception:
                iid = None
            trend = self.indicator_trend_by_id.get(iid) if iid is not None else None
            if trend:
                row_values[trend_idx] = trend

        ms_idx = idx_map.get("MARITAL_STATUS_ID")
        if ms_idx is not None and row_values[ms_idx] is not None and self.marital_by_id:
            try:
                mid = int(row_values[ms_idx])
            except Exception:
                mid = None
            m = self.marital_by_id.get(mid) if mid is not None else None
            if m:
                if "MARITAL_STATUS_EN" in idx_map:
                    row_values[idx_map["MARITAL_STATUS_EN"]] = m[0]
                if "MARITAL_STATUS_AR" in idx_map:
                    row_values[idx_map["MARITAL_STATUS_AR"]] = m[1]

        vt_idx = idx_map.get("VISIT_TYPE_ID")
        if vt_idx is not None and row_values[vt_idx] is not None and self.visit_type_by_id:
            try:
                vid = int(row_values[vt_idx])
            except Exception:
                vid = None
            vt = self.visit_type_by_id.get(vid) if vid is not None else None
            if vt:
                if "VISIT_TYPE_EN" in idx_map:
                    row_values[idx_map["VISIT_TYPE_EN"]] = vt[0]
                if "VISIT_TYPE_AR" in idx_map:
                    row_values[idx_map["VISIT_TYPE_AR"]] = vt[1]
                if "VISITORS_SUBCATEGORY_EN" in idx_map and (row_values[idx_map["VISITORS_SUBCATEGORY_EN"]] is None or PLACEHOLDER_RE.match(str(row_values[idx_map["VISITORS_SUBCATEGORY_EN"]]))):
                    row_values[idx_map["VISITORS_SUBCATEGORY_EN"]] = random.choice(self.column_domains.get("VISITORS_SUBCATEGORY", CURATED_DOMAINS["VISITORS_SUBCATEGORY"]))
                if "VISITORS_SUBCATEGORY_AR" in idx_map:
                    en = row_values[idx_map.get("VISITORS_SUBCATEGORY_EN")] if "VISITORS_SUBCATEGORY_EN" in idx_map else random.choice(self.column_domains.get("VISITORS_SUBCATEGORY", CURATED_DOMAINS["VISITORS_SUBCATEGORY"]))
                    row_values[idx_map["VISITORS_SUBCATEGORY_AR"]] = self.translate_en_to_ar(en)

        # Canonical status by STATUS_IN_EMPLOYMENT_ID
        sid_idx = idx_map.get("STATUS_IN_EMPLOYMENT_ID")
        if sid_idx is not None and row_values[sid_idx] is not None and self.status_by_id:
            try:
                sid = int(row_values[sid_idx])
            except Exception:
                sid = None
            status = self.status_by_id.get(sid) if sid is not None else None
            if status:
                en, ar = status
                if "STATUS_IN_EMPLOYMENT_EN" in idx_map:
                    row_values[idx_map["STATUS_IN_EMPLOYMENT_EN"]] = en
                if "STATUS_IN_EMPLOYMENT_AR" in idx_map:
                    row_values[idx_map["STATUS_IN_EMPLOYMENT_AR"]] = ar

        # Canonical property usage by PROPERTY_USAGE_ID
        pid_idx = idx_map.get("PROPERTY_USAGE_ID")
        if pid_idx is not None and row_values[pid_idx] is not None and self.property_usage_by_id:
            try:
                pid = int(row_values[pid_idx])
            except Exception:
                pid = None
            pu = self.property_usage_by_id.get(pid) if pid is not None else None
            if pu:
                en, ar = pu
                if "PROPERTY_USAGE_EN" in idx_map:
                    row_values[idx_map["PROPERTY_USAGE_EN"]] = en
                if "PROPERTY_USAGE_AR" in idx_map:
                    row_values[idx_map["PROPERTY_USAGE_AR"]] = ar

        # Canonical nationality/region by COUNTRY_ID when fields exist.
        cid_idx = idx_map.get("COUNTRY_ID")
        if cid_idx is not None and row_values[cid_idx] is not None and self.country_by_id:
            try:
                cid = int(row_values[cid_idx])
            except Exception:
                cid = None
            cmeta = self.country_by_id.get(cid) if cid is not None else None
            if cmeta:
                if "NATIONALITY" in idx_map:
                    row_values[idx_map["NATIONALITY"]] = cmeta["NATIONALITY_EN"]
                if "REGION" in idx_map:
                    row_values[idx_map["REGION"]] = cmeta["REGION_EN"]
                if "GENDER" in idx_map and row_values[idx_map["GENDER"]] not in ("Male", "Female"):
                    row_values[idx_map["GENDER"]] = random.choice(["Male", "Female"])

        # Canonical establishment sector text fields by ESTABLISHMENT_SECTOR_ID.
        sec_idx = idx_map.get("ESTABLISHMENT_SECTOR_ID")
        if sec_idx is not None and row_values[sec_idx] is not None and self.establishment_sector_by_id:
            try:
                sid = int(row_values[sec_idx])
            except Exception:
                sid = None
            sval = self.establishment_sector_by_id.get(sid) if sid is not None else None
            if sval:
                if "ESTABLISHMENT_SECTOR_EN" in idx_map:
                    row_values[idx_map["ESTABLISHMENT_SECTOR_EN"]] = sval[0]
                if "ESTABLISHMENT_SECTOR_AR" in idx_map:
                    row_values[idx_map["ESTABLISHMENT_SECTOR_AR"]] = sval[1]
                if "ACTIVITY_AR" in idx_map:
                    row_values[idx_map["ACTIVITY_AR"]] = SECTOR_ACTIVITY_AR.get(sval[0], sval[1])
                if "ESTABLISHMENT_NAME_AR" in idx_map:
                    row_values[idx_map["ESTABLISHMENT_NAME_AR"]] = f"مؤسسة {sval[1]}"

        # Trade type canonical values by TRADE_TYPE_ID.
        tidx = idx_map.get("TRADE_TYPE_ID")
        if tidx is not None and row_values[tidx] is not None:
            try:
                t_id = int(row_values[tidx])
            except Exception:
                t_id = None
            if t_id is not None:
                if t_id not in self.trade_type_by_id:
                    self.trade_type_by_id[t_id] = ("Import" if t_id % 2 else "Export", "استيراد" if t_id % 2 else "تصدير")
                t_en, t_ar = self.trade_type_by_id[t_id]
                if "TRADE_TYPE_EN" in idx_map:
                    row_values[idx_map["TRADE_TYPE_EN"]] = t_en
                if "TRADE_TYPE_AR" in idx_map:
                    row_values[idx_map["TRADE_TYPE_AR"]] = t_ar

        # Indicator name by ID for fact tables carrying text indicator labels.
        if iid_idx is not None and row_values[iid_idx] is not None:
            try:
                iid = int(row_values[iid_idx])
            except Exception:
                iid = None
            iname = self.indicator_name_by_id.get(iid) if iid is not None else None
            if iname:
                if "INDICATOR" in idx_map:
                    row_values[idx_map["INDICATOR"]] = iname

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
                    MONTH_NAME_AR.get(d.month, "عام"),
                    f"Q{quarter}",
                    f"الربع {quarter}",
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
                rows.append(
                    (
                        cid,
                        iso,
                        name,
                        nat,
                        ccode,
                        ccy,
                        region,
                        bool(gcc),
                        f"https://flags.example/{iso.lower()}.png",
                        self.translate_en_to_ar(name),
                        self.translate_en_to_ar(nat),
                        self.translate_en_to_ar(region),
                    )
                )
                self.country_by_id[int(cid)] = {
                    "COUNTRY_NAME_EN": name,
                    "NATIONALITY_EN": nat,
                    "REGION_EN": region,
                    "COUNTRY_NAME_AR": self.translate_en_to_ar(name),
                    "NATIONALITY_AR": self.translate_en_to_ar(nat),
                    "REGION_AR": self.translate_en_to_ar(region),
                }
            self.id_pools["COUNTRY_ID"] = [r[0] for r in rows]
            self.country_strength = {cid: np.clip(np.random.normal(1.0 + (0.3 if cid == 634 else 0), 0.2), 0.5, 1.7) for cid in self.id_pools["COUNTRY_ID"]}
            return rows

        if table == "dim_gender":
            rows = [
                (1, 1, 1, 1, "Male", "ذكر", "run-1", "trg-1", dt.datetime.utcnow(), "", "pipeline", "2026"),
                (2, 2, 2, 2, "Female", "أنثى", "run-1", "trg-1", dt.datetime.utcnow(), "", "pipeline", "2026"),
            ]
            self.id_pools["GENDER_ID"] = [1, 2]
            return [tuple(r[: len(cols)]) for r in rows]

        if table == "dim_marital_status":
            rows = [(1, "Single", "أعزب"), (2, "Married", "متزوج"), (3, "Divorced", "مطلق")]
            self.id_pools["MARITAL_STATUS_ID"] = [r[0] for r in rows]
            self.marital_by_id = {int(r[0]): (r[1], r[2]) for r in rows}
            return rows

        if table == "hmc_dim_visit_type":
            rows = [(1, "Outpatient", "عيادات خارجية"), (2, "Emergency", "طوارئ"), (3, "Inpatient", "تنويم")]
            self.id_pools["VISIT_TYPE_ID"] = [r[0] for r in rows]
            self.visit_type_by_id = {int(r[0]): (r[1], r[2]) for r in rows}
            return rows

        if table == "cgb_dim_status_in_employment":
            base = self.column_domains.get("STATUS_IN_EMPLOYMENT", CURATED_DOMAINS["STATUS_IN_EMPLOYMENT"])
            rows = [(i + 1, v, self.translate_en_to_ar(v)) for i, v in enumerate(base[: max(1, count)])]
            self.id_pools["STATUS_IN_EMPLOYMENT_ID"] = [r[0] for r in rows]
            self.status_by_id = {int(r[0]): (r[1], r[2]) for r in rows}
            return rows

        if table == "moj_dim_property_usage":
            base = self.column_domains.get("PROPERTY_USAGE", CURATED_DOMAINS["PROPERTY_USAGE"])
            rows = [(i + 1, v, self.translate_en_to_ar(v)) for i, v in enumerate(base[: max(1, count)])]
            self.id_pools["PROPERTY_USAGE_ID"] = [r[0] for r in rows]
            self.property_usage_by_id = {int(r[0]): (r[1], r[2]) for r in rows}
            return rows

        if table == "dim_entities":
            base = self.column_domains.get("ENTITY", CURATED_DOMAINS["ENTITY"])
            rows = [(i, base[(i - 1) % len(base)]) for i in range(1, max(3, count) + 1)]
            self.id_pools["ENTITY_ID"] = [r[0] for r in rows]
            return rows

        if table == "dim_establishment_sector":
            base = self.column_domains.get("SECTOR", CURATED_DOMAINS["ACTIVITY_TYPE"])
            rows = []
            for i in range(1, max(3, count) + 1):
                sector = base[(i - 1) % len(base)]
                sector_ar = self.translate_en_to_ar(sector)
                rows.append((i, i, i, i, sector, sector_ar, f"run-{i}", f"trg-{i}", dt.datetime.utcnow(), f"run-{i}", "pipeline", f"{self.pick_year()}-{int(np.random.randint(1,13)):02d}"))
                self.establishment_sector_by_id[i] = (sector, sector_ar)
            self.id_pools["ESTABLISHMENT_SECTOR_ID"] = [r[3] for r in rows]
            return [tuple(r[: len(cols)]) for r in rows]

        if table == "dim_education_level":
            lvl0 = ["Primary", "Intermediate", "Secondary", "Post-secondary"]
            lvl1 = ["Academic", "Technical", "Vocational", "University"]
            skill = ["Foundational", "Applied", "Advanced"]
            rows = []
            for i in range(1, max(4, count) + 1):
                e0 = lvl0[(i - 1) % len(lvl0)]
                e1 = lvl1[(i - 1) % len(lvl1)]
                sk = skill[(i - 1) % len(skill)]
                rows.append((i, i, i, i, e0, self.translate_en_to_ar(e0), e1, self.translate_en_to_ar(e1), sk, self.translate_en_to_ar(sk), f"run-{i}", f"trg-{i}", dt.datetime.utcnow(), f"run-{i}", "pipeline", f"{self.pick_year()}-{int(np.random.randint(1,13)):02d}"))
            self.id_pools["EDUCATION_LEVEL_ID"] = [r[3] for r in rows]
            return [tuple(r[: len(cols)]) for r in rows]

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
                sector = (m.get("Indicator Sector (cleansed)") if isinstance(m, dict) else None) or random.choice(self.column_domains.get("SECTOR", CURATED_DOMAINS["ACTIVITY_TYPE"]))
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
                    elif c == "CALCULATION_METHODOLOGY":
                        row[j] = "Administrative calculation methodology"
                    elif c == "SOURCE_TYPE":
                        row[j] = random.choice(["Administrative Data", "Survey Data", "International Database"])
                    elif c == "FREQUENCY_OF_MEASUREMENT":
                        row[j] = random.choice(self.column_domains.get("FREQUENCY", ["Annual", "Quarterly", "Monthly"]))
                    elif c == "INDICATOR_SOURCE_ENTITY":
                        row[j] = random.choice(["National Statistics Authority", "World Bank", "International Monetary Fund", "Ministry of Planning"])
                    elif c == "OVERLAP_WITH_NDS3":
                        row[j] = random.choice(["Yes", "No", "Partial"])
                    elif c == "LINK_TO_SOURCE":
                        row[j] = f"https://data.example/indicator/{iid}"
                    elif c == "PILLAR_MAPPING":
                        row[j] = random.choice(["Economic Pillar", "Social Pillar", "Environmental Pillar", "Institutional Pillar"])
                    elif c == "OUTCOME_MAPPING":
                        row[j] = random.choice(["Productivity Outcome", "Inclusion Outcome", "Sustainability Outcome"])
                    elif c == "NATIONAL_DEVELOPMENT_STRATEGY_SECTOR":
                        row[j] = sector
                    elif c == "NATIONAL_DEVELOPMENT_STRATEGY_CLUSTER":
                        row[j] = random.choice(["Human Capital", "Economic Diversification", "Sustainable Development"])
                    elif c == "BASELINE_VALUE":
                        row[j] = f"{round(np.random.uniform(10, 80), 2)}"
                    elif c == "TARGET_VALUE":
                        row[j] = f"{round(np.random.uniform(20, 95), 2)}"
                    elif c == "INDICATOR_MEASUREMENT_UNIT":
                        row[j] = random.choice(["Percent", "Index", "Ranking", "QAR Million", "Hours"])
                    elif c == "TARGET_VALUE_CLEANSED":
                        row[j] = f"{round(np.random.uniform(20, 95), 2)}"
                    elif c == "LATEST_VALUE_MEASUREMENT_UNIT":
                        row[j] = random.choice(["Percent", "Index", "Ranking", "QAR Million", "Hours"])
                    elif c == "LATEST_VALUE":
                        row[j] = f"{round(np.random.uniform(10, 95), 2)}"
                    elif c == "TREND":
                        row[j] = random.choice(["Improving", "Stable", "Declining"])
                    elif c == "QDTI_SCORE":
                        row[j] = f"{round(np.random.uniform(0, 100), 2)}"
                    elif c == "INFOBOX_TOOLTIP":
                        row[j] = "Indicator metadata tooltip"
                    elif c == "SECOND_LATEST_DATE":
                        row[j] = str(self.pick_year() - 1)
                    elif c == "SECOND_LATEST_VALUE":
                        row[j] = f"{round(np.random.uniform(10, 95), 2)}"
                    elif c == "LATEST_DATE_WHERE_QATAR_IS_AVAILABLE":
                        row[j] = str(self.args.end_year)
                    elif c == "QDTI_DESCRIPTION_EN":
                        row[j] = f"Detailed description for {name}"
                    elif c == "QDTI_DESCRIPTION_AR":
                        name_ar = self.ensure_arabic_text(self.translate_en_to_ar(name))
                        row[j] = f"وصف تفصيلي لـ {name_ar}"
                    elif c == "INDICATOR_NAME_AR":
                        row[j] = self.ensure_arabic_text(self.translate_en_to_ar(name))
                    elif c == "INDICATOR_DEFINITION_AR":
                        row[j] = self.ensure_arabic_text(self.metric_get(m, "Short Description AR", "Short Description AR ") or self.translate_en_to_ar(definition))
                    elif c == "INDICATOR_SECTOR_AR":
                        row[j] = self.ensure_arabic_text(self.translate_en_to_ar(sector))
                    elif c == "EARLIEST_DATE_WHERE_QATAR_IS_AVAILABLE":
                        row[j] = self.args.start_year
                    elif c == "TARGET_VALUE_NUMERIC":
                        row[j] = round(np.random.uniform(10, 90), 3)
                    elif c == "LAST_UPDATED":
                        row[j] = dt.datetime.utcnow()
                    elif c == "TARGET_YEAR":
                        row[j] = self.args.end_year
                    elif c == "TREND_DIRECTION":
                        row[j] = random.choice(["Up", "Down", "Stable"])
                    elif c.endswith("_AR"):
                        row[j] = self.translate_en_to_ar(row[j - 1] if j > 0 else name)
                    elif c in ("IS_QPULSE",):
                        row[j] = bool(np.random.rand() > 0.5)
                    elif dtype in ("LONG", "INT"):
                        row[j] = int(max(0, round(np.random.normal(50, 20))))
                    elif dtype in ("DOUBLE", "FLOAT", "DECIMAL"):
                        row[j] = round(float(np.random.uniform(0, 100)), 4)
                    else:
                        row[j] = c.replace("_", " ").title()
                self.enforce_ar_pairs_for_row(table, cols, row, metric_hint=m if isinstance(m, dict) else None)
                rows.append(tuple(row))
                self.indicator_name_by_id[int(iid)] = str(name)
                if "INDICATOR_ID" in cols and "TREND_DIRECTION" in cols:
                    iid = row[cols.index("INDICATOR_ID")]
                    tr = row[cols.index("TREND_DIRECTION")]
                    if iid is not None and tr is not None:
                        self.indicator_trend_by_id[int(iid)] = str(tr)
            self.id_pools["INDICATOR_ID"] = [r[cols.index("INDICATOR_ID")] for r in rows if "INDICATOR_ID" in cols]
            return rows

        rows = []
        key_col = detect_key_column(self.tables[table])
        for i in range(1, count + 1):
            row = []
            for c in cols:
                row.append(self.make_value(table, c, i, {}))
            self.enforce_ar_pairs_for_row(table, cols, row)
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
        if col in self.translation_protected_cols.get(table, set()):
            return False
        if self.tables[table].dtype(col) == "STRING" and self.is_controlled_column(col):
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

        # Controlled columns: no free random word fallback.
        if dtype == "STRING" and self.is_controlled_column(cname):
            forced = self.controlled_text_value(table, cname, row_ctx)
            if forced is not None:
                row_ctx[cname] = forced
                return forced
            if cname.endswith("_AR"):
                en_col = cname[:-3] + "_EN"
                base_col = cname[:-3]
                en_val = row_ctx.get(en_col) or row_ctx.get(base_col)
                if en_val is None:
                    en_val = self.default_english_for_pair(en_col if en_col else base_col)
                    row_ctx[en_col if en_col else base_col] = en_val
                return self.translate_en_to_ar(en_val)
            if any(k in cname for k in ["RATIO", "SCORE", "RANK", "GDP_PER_CAPITA", "POPULATION", "VALUE"]):
                num = self.numeric_value(cname, row_ctx.get("COUNTRY_ID"), row_ctx.get("YEAR"))
                if "RANK" in cname:
                    return str(int(max(1, round(num))))
                return f"{float(num):.2f}"
            fallback = self.default_english_for_pair(cname)
            row_ctx[cname] = fallback
            return fallback

        if dtype in ("LONG", "INT"):
            if "RANK" in cname:
                return int(np.clip(np.random.normal(60, 30), 1, 250))
            if "QUARTER" in cname:
                return int(np.random.randint(1, 5))
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
            return random.choice(self.translation_safe_choices["COUNTRY_NAME"])
        if "NATIONALITY" in cname:
            return random.choice(self.translation_safe_choices["NATIONALITY"])
        if "CURRENCY_CODE" in cname:
            return random.choice(["QAR", "USD", "EUR", "SAR", "AED"])
        if "TREND_DIRECTION" in cname:
            return random.choice(["Up", "Down", "Stable"])
        if table == "intl_fact_pisa_ranking" and cname == "READING_MEAN":
            return round(float(np.random.uniform(320, 620)), 2)
        if table == "intl_fact_out_of_pocket_expenditure" and cname == "INDICATOR":
            iid = row_ctx.get("INDICATOR_ID")
            if iid is not None:
                try:
                    return self.indicator_name_by_id.get(int(iid), "Out-of-pocket Expenditure")
                except Exception:
                    return "Out-of-pocket Expenditure"
            return "Out-of-pocket Expenditure"
        if table.startswith("hmc_fact_") and cname == "VISITORS_SUBCATEGORY_EN":
            return random.choice(self.column_domains.get("VISITORS_SUBCATEGORY", CURATED_DOMAINS["VISITORS_SUBCATEGORY"]))
        if table.startswith("hmc_fact_") and cname == "VISITORS_SUBCATEGORY_AR":
            en_val = row_ctx.get("VISITORS_SUBCATEGORY_EN") or random.choice(self.column_domains.get("VISITORS_SUBCATEGORY", CURATED_DOMAINS["VISITORS_SUBCATEGORY"]))
            return self.translate_en_to_ar(en_val)
        if table == "intl_fact_ntm_frequency_ratio" and cname == "TRADE_TYPE_EN":
            t_id = row_ctx.get("TRADE_TYPE_ID")
            if t_id is None:
                return random.choice(self.column_domains.get("TRADE_TYPE", CURATED_DOMAINS["TRADE_TYPE"]))
            try:
                t_idi = int(t_id)
            except Exception:
                t_idi = 1
            return self.trade_type_by_id.get(t_idi, ("Import", "استيراد"))[0]
        if table == "intl_fact_ntm_frequency_ratio" and cname == "TRADE_TYPE_AR":
            t_id = row_ctx.get("TRADE_TYPE_ID")
            if t_id is None:
                return self.translate_en_to_ar(random.choice(self.column_domains.get("TRADE_TYPE", CURATED_DOMAINS["TRADE_TYPE"])))
            try:
                t_idi = int(t_id)
            except Exception:
                t_idi = 1
            return self.trade_type_by_id.get(t_idi, ("Import", "استيراد"))[1]
        if table == "mof_fact_national_budget" and cname == "CATEGORY_EN":
            return random.choice(self.column_domains.get("BUDGET_CATEGORY", CURATED_DOMAINS["BUDGET_CATEGORY"]))
        if table == "mof_fact_national_budget" and cname == "CATEGORY_AR":
            en = row_ctx.get("CATEGORY_EN") or random.choice(self.column_domains.get("BUDGET_CATEGORY", CURATED_DOMAINS["BUDGET_CATEGORY"]))
            return self.translate_en_to_ar(en)
        if table == "mof_fact_national_budget" and cname == "ITEM_EN":
            cat = row_ctx.get("CATEGORY_EN")
            if cat in BUDGET_ITEMS_BY_CATEGORY:
                return random.choice(BUDGET_ITEMS_BY_CATEGORY[str(cat)])
            return random.choice(self.column_domains.get("ITEM", CURATED_DOMAINS["TREASURY_ITEM"]))
        if table == "mof_fact_national_budget" and cname == "ITEM_AR":
            en = row_ctx.get("ITEM_EN") or random.choice(self.column_domains.get("ITEM", CURATED_DOMAINS["TREASURY_ITEM"]))
            return self.translate_en_to_ar(en)
        if table == "mof_fact_return_on_treasury" and cname == "ITEM_EN":
            return random.choice(self.column_domains.get("TREASURY_ITEM", CURATED_DOMAINS["TREASURY_ITEM"]))
        if table == "mof_fact_return_on_treasury" and cname == "ITEM_AR":
            en = row_ctx.get("ITEM_EN") or random.choice(self.column_domains.get("TREASURY_ITEM", CURATED_DOMAINS["TREASURY_ITEM"]))
            return self.translate_en_to_ar(en)
        if table == "grsia_fact_grsia_data" and cname == "EMPLOYER_AR":
            return random.choice(["شركة وطنية", "مؤسسة قطرية", "جهة حكومية", "شركة خاصة"])
        if table == "grsia_fact_grsia_data" and cname == "ACTIVITY_AR":
            return random.choice(list(SECTOR_ACTIVITY_AR.values()))
        if table == "grsia_fact_grsia_data" and cname == "ESTABLISHMENT_NAME_AR":
            return random.choice(["مؤسسة الدوحة", "شركة الخليج", "مجموعة قطر", "مركز الخدمات"])
        if table.startswith("moehe_fact_") and cname == "ACADEMIC_STAGE":
            lvl = row_ctx.get("ACADEMIC_LEVEL_ID")
            if lvl is not None:
                try:
                    lid = int(lvl)
                    return f"Grade {((lid - 1) % 12) + 1}"
                except Exception:
                    pass
            return random.choice(self.column_domains.get("ACADEMIC_STAGE", CURATED_DOMAINS["ACADEMIC_STAGE"]))
        if table == "gac_fact_time_to_import_and_export" and cname == "CATEGORY_EN":
            val = random.choice(["Export", "Import"])
            row_ctx[cname] = val
            return val
        if table == "gac_fact_time_to_import_and_export" and cname == "CATEGORY_AR":
            en_val = row_ctx.get("CATEGORY_EN")
            if en_val not in ("Export", "Import"):
                en_val = random.choice(["Export", "Import"])
                row_ctx["CATEGORY_EN"] = en_val
            return EN_AR_EXACT_MAP.get(str(en_val), self.translate_en_to_ar(en_val))
        if "CATEGORY" in cname or "TYPE" in cname or "GROUP" in cname or "SECTOR" in cname:
            return self.default_english_for_pair(cname)
        if cname.endswith("_AR"):
            en_col = cname[:-3] + "_EN"
            en_val = row_ctx.get(en_col)
            return self.translate_en_to_ar(en_val) if en_val is not None else "عام"
        if cname.endswith("_EN"):
            return self.default_english_for_pair(cname)
        if "PIPELINE" in cname:
            return random.choice(["npc_pipeline", "gold_pipeline", "metrics_pipeline"])
        if cname == "ADF_PERIOD":
            return f"{self.pick_year()}-{int(np.random.randint(1,13)):02d}"
        if "URL" in cname:
            return f"https://example.com/{self.fake.slug()}"
        if "ACADEMIC_YEAR" in cname:
            y = row_ctx.get("YEAR")
            if y is None:
                y = self.pick_year()
            return f"{y}/{y+1}"
        if "QUARTER" in cname:
            return random.choice(["Q1", "Q2", "Q3", "Q4"])
        if "ITEM" in cname:
            return self.default_english_for_pair(cname)
        if dtype == "STRING":
            if "DESCRIPTION" in cname or "DEFINITION" in cname:
                return "General description"
            if cname == "INDICATOR":
                iid = row_ctx.get("INDICATOR_ID")
                if iid is not None:
                    try:
                        return self.indicator_name_by_id.get(int(iid), "Indicator")
                    except Exception:
                        return "Indicator"
                return "Indicator"
            label = cname.replace("_", " ").title()
            return f"{label} {idx}"
        return self.fake.word().title()

    def generate_fact_rows(self, table: str, count: int) -> Iterable[Tuple]:
        cols = self.tables[table].column_names
        ycol = table_year_column(cols)
        date_idx = cols.index("DATE_ID") if "DATE_ID" in cols else None
        year_idx = cols.index("YEAR") if "YEAR" in cols else None
        survey_year_idx = cols.index("SURVEY_YEAR") if "SURVEY_YEAR" in cols else None
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
            self.enforce_ar_pairs_for_row(table, cols, row)
            self.enforce_id_consistency_for_row(table, cols, row)

            # Enforce strict DATE_ID -> YEAR/SURVEY_YEAR consistency.
            if date_idx is not None and row[date_idx] is not None:
                date_year = int(str(row[date_idx])[:4])
                if year_idx is not None:
                    row[year_idx] = date_year
                if survey_year_idx is not None:
                    row[survey_year_idx] = date_year
            # Ensure ACADEMIC_YEAR contains YEAR when both exist.
            ay_idx = cols.index("ACADEMIC_YEAR") if "ACADEMIC_YEAR" in cols else None
            if ay_idx is not None:
                yr_val = row[year_idx] if year_idx is not None else ctx.get("YEAR")
                if yr_val is not None:
                    try:
                        y = int(yr_val)
                        row[ay_idx] = f"{y}/{y+1}"
                    except Exception:
                        pass

            if has_triplet:
                yv = ctx.get(ycol) if ycol else None
                if date_idx is not None and row[date_idx] is not None and ycol in ("YEAR", "SURVEY_YEAR"):
                    yv = int(str(row[date_idx])[:4])
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


def validate_source_of_truth(cur, tables: Dict[str, TableMeta], fk_edges: Dict[str, List[Tuple[str, str, str]]], gen: "SyntheticGenerator") -> List[str]:
    issues: List[str] = []

    # 1) _AR completeness where source EN/base exists.
    for tname, tmeta in tables.items():
        cols = tmeta.column_names
        col_set = set(cols)
        for c in cols:
            if not c.endswith("_AR"):
                continue
            src = c[:-3] + "_EN" if (c[:-3] + "_EN") in col_set else (c[:-3] if c[:-3] in col_set else None)
            if not src:
                continue
            sql = (
                f"SELECT COUNT(*) FROM {qident(tname)} WHERE {qident(src)} IS NOT NULL "
                f"AND TRIM(CAST({qident(src)} AS CHAR)) <> '' "
                f"AND ({qident(c)} IS NULL OR TRIM(CAST({qident(c)} AS CHAR)) = '')"
            )
            cur.execute(sql)
            bad = int(cur.fetchone()[0])
            if bad > 0:
                issues.append(f"{tname}.{c}: {bad} missing AR values where {src} exists")

    # 2) Key-linked shared-column consistency.
    for child, refs in fk_edges.items():
        child_cols = set(tables[child].column_names)
        for ccol, parent, pcol in refs:
            parent_cols = set(tables[parent].column_names)
            shared = [c for c in child_cols.intersection(parent_cols) if c not in {ccol, pcol}]
            for col in shared:
                if col in {"PIPELINE_RUN_ID", "PIPELINE_TRIGGER_ID", "PIPELINE_TRIGGER_TIME", "PIPELINE_TRIGGERED_BY_PIPELINE_RUN_ID", "PIPELINE_NAME", "ADF_PERIOD"}:
                    continue
                sql = (
                    f"SELECT COUNT(*) FROM {qident(child)} c "
                    f"JOIN {qident(parent)} p ON c.{qident(ccol)} = p.{qident(pcol)} "
                    f"WHERE c.{qident(col)} IS NOT NULL AND p.{qident(col)} IS NOT NULL "
                    f"AND CAST(c.{qident(col)} AS CHAR) <> CAST(p.{qident(col)} AS CHAR)"
                )
                cur.execute(sql)
                bad = int(cur.fetchone()[0])
                if bad > 0:
                    issues.append(f"{child}.{col} mismatch vs {parent}.{col} via {ccol}: {bad} rows")

    # 3) Domain checks for strict controlled columns.
    checks = [
        ("cgb_dim_status_in_employment", "STATUS_IN_EMPLOYMENT_EN", "STATUS_IN_EMPLOYMENT"),
        ("moj_dim_property_usage", "PROPERTY_USAGE_EN", "PROPERTY_USAGE"),
        ("hmc_fact_hmc_revenue", "VISITORS_SUBCATEGORY_EN", "VISITORS_SUBCATEGORY"),
        ("hmc_fact_hmc_visitors", "VISITORS_SUBCATEGORY_EN", "VISITORS_SUBCATEGORY"),
        ("intl_fact_ntm_frequency_ratio", "TRADE_TYPE_EN", "TRADE_TYPE"),
        ("intl_fact_network_readiness_index_ranking", "INCOME_GROUP", "INCOME_GROUP"),
        ("intl_fact_msci_all_country_world_index", "MARKET_TYPE", "MARKET_TYPE"),
        ("moj_fact_real_estate_transactions", "PROPERTY_TYPE", "PROPERTY_TYPE"),
        ("cgb_fact_cgb_employee", "MAIN_OCCUPATION", "OCCUPATION"),
        ("mof_fact_national_budget", "CATEGORY_EN", "BUDGET_CATEGORY"),
        ("mof_fact_national_budget", "ITEM_EN", "ITEM"),
        ("mof_fact_return_on_treasury", "ITEM_EN", "TREASURY_ITEM"),
        ("dim_entities", "ENTITY_EN", "ENTITY"),
    ]
    for tname, col, dom in checks:
        if tname not in tables or col not in set(tables[tname].column_names):
            continue
        allowed = gen.column_domains.get(dom, [])
        if not allowed:
            continue
        vals = ", ".join(mysql_literal(v) for v in allowed)
        sql = (
            f"SELECT COUNT(*) FROM {qident(tname)} WHERE {qident(col)} IS NOT NULL "
            f"AND CAST({qident(col)} AS CHAR) NOT IN ({vals})"
        )
        cur.execute(sql)
        bad = int(cur.fetchone()[0])
        if bad > 0:
            issues.append(f"{tname}.{col}: {bad} values outside domain {dom}")

    # 4) Quarter constraints.
    if "mof_fact_return_on_treasury" in tables and "QUARTER" in set(tables["mof_fact_return_on_treasury"].column_names):
        sql = (
            f"SELECT COUNT(*) FROM {qident('mof_fact_return_on_treasury')} "
            f"WHERE {qident('QUARTER')} IS NOT NULL "
            f"AND ({qident('QUARTER')} < 1 OR {qident('QUARTER')} > 4)"
        )
        cur.execute(sql)
        bad = int(cur.fetchone()[0])
        if bad > 0:
            issues.append(f"mof_fact_return_on_treasury.QUARTER: {bad} out of range [1,4]")

    # 5) Placeholder token detector.
    for tname, tmeta in tables.items():
        for col in tmeta.column_names:
            if tmeta.dtype(col) != "STRING":
                continue
            sql = (
                f"SELECT COUNT(*) FROM {qident(tname)} WHERE {qident(col)} IS NOT NULL AND ("
                f"CAST({qident(col)} AS CHAR) GLOB 'Status [0-9]*' OR "
                f"CAST({qident(col)} AS CHAR) GLOB 'Usage [0-9]*' OR "
                f"CAST({qident(col)} AS CHAR) GLOB 'Type [A-Z]*' OR "
                f"CAST({qident(col)} AS CHAR) GLOB 'Category [A-Z]*' OR "
                f"CAST({qident(col)} AS CHAR) GLOB 'Item [0-9]*' OR "
                f"CAST({qident(col)} AS CHAR) GLOB 'Indicator [0-9]*' OR "
                f"CAST({qident(col)} AS CHAR) GLOB 'Academic Stage [0-9]*')"
            )
            cur.execute(sql)
            bad = int(cur.fetchone()[0])
            if bad > 0:
                cur.execute(
                    f"SELECT CAST({qident(col)} AS CHAR) FROM {qident(tname)} WHERE {qident(col)} IS NOT NULL AND ("
                    f"CAST({qident(col)} AS CHAR) GLOB 'Status [0-9]*' OR "
                    f"CAST({qident(col)} AS CHAR) GLOB 'Usage [0-9]*' OR "
                    f"CAST({qident(col)} AS CHAR) GLOB 'Type [A-Z]*' OR "
                    f"CAST({qident(col)} AS CHAR) GLOB 'Category [A-Z]*' OR "
                    f"CAST({qident(col)} AS CHAR) GLOB 'Item [0-9]*' OR "
                    f"CAST({qident(col)} AS CHAR) GLOB 'Indicator [0-9]*' OR "
                    f"CAST({qident(col)} AS CHAR) GLOB 'Academic Stage [0-9]*') LIMIT 1"
                )
                sample = cur.fetchone()
                sval = sample[0] if sample else ""
                issues.append(f"{tname}.{col}: {bad} placeholder-pattern values (sample='{sval}')")

    # 6) Generic AR filler overuse and low-information AR values.
    for tname, tmeta in tables.items():
        cols = tmeta.column_names
        col_set = set(cols)
        for c in cols:
            if not c.endswith("_AR"):
                continue
            src = c[:-3] + "_EN" if (c[:-3] + "_EN") in col_set else (c[:-3] if c[:-3] in col_set else None)
            if src:
                sql = (
                    f"SELECT COUNT(*) FROM {qident(tname)} WHERE {qident(src)} IS NOT NULL "
                    f"AND TRIM(CAST({qident(src)} AS CHAR)) <> '' "
                    f"AND CAST({qident(src)} AS CHAR) NOT IN ('General','Public') "
                    f"AND CAST({qident(src)} AS CHAR) GLOB '*[A-Za-z]*' "
                    f"AND CAST({qident(c)} AS CHAR) IN ('عام','فئة','نوع','قيمة')"
                )
                cur.execute(sql)
                bad = int(cur.fetchone()[0])
                if bad > 0:
                    issues.append(f"{tname}.{c}: {bad} low-information AR translations (source={src})")
            cur.execute(f"SELECT COUNT(*), COUNT(DISTINCT CAST({qident(c)} AS CHAR)) FROM {qident(tname)} WHERE {qident(c)} IS NOT NULL")
            total, distinct_n = cur.fetchone()
            total = int(total)
            distinct_n = int(distinct_n)
            if total >= 10 and distinct_n <= 1:
                cur.execute(f"SELECT CAST({qident(c)} AS CHAR) FROM {qident(tname)} WHERE {qident(c)} IS NOT NULL LIMIT 1")
                topv = cur.fetchone()
                topv = str(topv[0]) if topv else ""
                if topv in ARABIC_FILLER_VALUES:
                    issues.append(f"{tname}.{c}: dominant generic AR filler '{topv}' across {total} rows")

    # 7) Semantic type checks for numeric-semantic fields stored as text.
    if "intl_fact_pisa_ranking" in tables and "READING_MEAN" in set(tables["intl_fact_pisa_ranking"].column_names):
        sql = (
            f"SELECT COUNT(*) FROM {qident('intl_fact_pisa_ranking')} "
            f"WHERE {qident('READING_MEAN')} IS NOT NULL AND CAST({qident('READING_MEAN')} AS CHAR) GLOB '*[A-Za-z]*'"
        )
        cur.execute(sql)
        bad = int(cur.fetchone()[0])
        if bad > 0:
            issues.append(f"intl_fact_pisa_ranking.READING_MEAN: {bad} non-numeric placeholder values")

    # 8) ACADEMIC_YEAR must contain YEAR where both fields exist.
    for tname, tmeta in tables.items():
        cols = set(tmeta.column_names)
        if "ACADEMIC_YEAR" in cols and "YEAR" in cols:
            sql = (
                f"SELECT COUNT(*) FROM {qident(tname)} WHERE {qident('YEAR')} IS NOT NULL AND {qident('ACADEMIC_YEAR')} IS NOT NULL "
                f"AND INSTR(CAST({qident('ACADEMIC_YEAR')} AS CHAR), CAST({qident('YEAR')} AS CHAR)) = 0"
            )
            cur.execute(sql)
            bad = int(cur.fetchone()[0])
            if bad > 0:
                issues.append(f"{tname}.ACADEMIC_YEAR: {bad} rows missing YEAR token")

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


def validate_sql_file_content(sql_file: Path, tables: Dict[str, TableMeta], fk_edges: Dict[str, List[Tuple[str, str, str]]], gen: "SyntheticGenerator") -> List[str]:
    import sqlite3

    raw = sql_file.read_text(encoding="utf-8")
    s = raw.replace("\r", "")
    s = re.sub(r"^\s*SET\s+.*?;\s*$", "", s, flags=re.M | re.I)
    s = re.sub(r"^\s*CREATE\s+DATABASE\s+.*?;\s*$", "", s, flags=re.M | re.I)
    s = re.sub(r"^\s*USE\s+`?\w+`?;\s*$", "", s, flags=re.M | re.I)
    s = re.sub(r"^\s*START\s+TRANSACTION\s*;\s*$", "", s, flags=re.M | re.I)
    s = re.sub(r"^\s*COMMIT\s*;\s*$", "", s, flags=re.M | re.I)
    con = sqlite3.connect(":memory:")
    cur = con.cursor()
    cur.executescript(s.replace("`", "\""))
    return validate_source_of_truth(cur, tables, fk_edges, gen)


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
        if not args.skip_validation and args.strict_quality:
            source_issues = validate_sql_file_content(out_path, tables, fk_edges, gen)
            if source_issues:
                logging.error("Source-of-truth validation issues: %d", len(source_issues))
                for x in source_issues[:40]:
                    logging.error("%s", x)
                raise RuntimeError("Source-of-truth validation failed for generated SQL file")
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
            source_issues = validate_source_of_truth(cur, tables, fk_edges, gen) if args.strict_quality else []

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
            if source_issues:
                logging.error("Source-of-truth issues: %d", len(source_issues))
                for x in source_issues[:40]:
                    logging.error("%s", x)
                raise RuntimeError("Source-of-truth validation failed")

        cur.close()
        conn.close()

    total_time = time.time() - t0
    logging.info("Generation completed in %.2fs", total_time)
    print("\n=== TABLE ROW COUNTS ===")
    for t in sorted(inserted):
        print(f"{t}: {inserted[t]}")

if __name__ == "__main__":
    main()

"""Deterministic preprocessing for the UCI Online Retail II workbook."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


REQUIRED_COLUMNS = {
    "invoice": ("Invoice", "InvoiceNo"),
    "stock_code": ("StockCode",),
    "quantity": ("Quantity",),
    "invoice_date": ("InvoiceDate",),
    "unit_price": ("Price", "UnitPrice"),
    "customer_id": ("Customer ID", "CustomerID"),
    "country": ("Country",),
}


@dataclass(frozen=True, order=True)
class RetailEvent:
    customer_id: str
    invoice: str
    timestamp: datetime
    mark: int
    is_cancellation: bool
    abs_quantity: float
    line_count: int


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _customer(value: Any) -> str | None:
    if _missing(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    return text or None


def _resolve_columns(row: Mapping[str, Any]) -> dict[str, str]:
    columns = set(row)
    resolved: dict[str, str] = {}
    for canonical, aliases in REQUIRED_COLUMNS.items():
        match = next((alias for alias in aliases if alias in columns), None)
        if match is None:
            raise ValueError(f"missing required Online Retail II column: {aliases}")
        resolved[canonical] = match
    return resolved


def _timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, np.datetime64):
        seconds = value.astype("datetime64[us]").astype(int) / 1_000_000
        return datetime.fromtimestamp(seconds)
    try:
        return datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"invalid InvoiceDate value: {value!r}") from exc


def _dedup_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(f"{key}={row[key]!r}" for key in sorted(row))


def preprocess_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    quantity_threshold: float = 20.0,
    min_customer_events: int = 5,
) -> tuple[list[RetailEvent], dict[str, Any]]:
    """Remove exact duplicate lines, aggregate invoices, and assign frozen marks.

    Marks are outcome-independent: 0 cancellation, 1 purchase with aggregate
    absolute quantity at most the frozen threshold, and 2 larger purchase.
    """

    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("Online Retail II rows must not be empty")
    if not math.isfinite(quantity_threshold) or quantity_threshold <= 0:
        raise ValueError("quantity_threshold must be finite and positive")
    if not isinstance(min_customer_events, int) or min_customer_events < 1:
        raise ValueError("min_customer_events must be a positive integer")
    columns = _resolve_columns(materialized[0])
    unique_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    duplicate_count = 0
    for row in materialized:
        _resolve_columns(row)
        key = _dedup_key(row)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        unique_rows.append(row)

    missing_customer = 0
    grouped: dict[tuple[str, str, datetime], list[Mapping[str, Any]]] = defaultdict(list)
    for row in unique_rows:
        customer = _customer(row[columns["customer_id"]])
        if customer is None:
            missing_customer += 1
            continue
        invoice = str(row[columns["invoice"]]).strip()
        when = _timestamp(row[columns["invoice_date"]])
        grouped[(customer, invoice, when)].append(row)

    all_events: list[RetailEvent] = []
    for (customer, invoice, when), lines in grouped.items():
        quantities = [float(line[columns["quantity"]]) for line in lines]
        cancelled = invoice.upper().startswith("C") or any(value < 0 for value in quantities)
        abs_quantity = float(sum(abs(value) for value in quantities))
        mark = 0 if cancelled else 1 if abs_quantity <= quantity_threshold else 2
        all_events.append(
            RetailEvent(
                customer_id=customer,
                invoice=invoice,
                timestamp=when,
                mark=mark,
                is_cancellation=cancelled,
                abs_quantity=abs_quantity,
                line_count=len(lines),
            )
        )
    counts = Counter(event.customer_id for event in all_events)
    eligible = {customer for customer, count in counts.items() if count >= min_customer_events}
    events = sorted(event for event in all_events if event.customer_id in eligible)
    timestamp_counts = Counter((event.customer_id, event.timestamp) for event in events)
    profile = {
        "raw_rows": len(materialized),
        "exact_duplicate_rows": duplicate_count,
        "unique_rows": len(unique_rows),
        "missing_customer_rows": missing_customer,
        "aggregated_invoices_before_customer_filter": len(all_events),
        "processed_events": len(events),
        "customers_before_min_event_filter": len(counts),
        "eligible_customers": len(eligible),
        "min_customer_events": min_customer_events,
        "quantity_threshold": float(quantity_threshold),
        "cancellation_events": sum(event.is_cancellation for event in events),
        "mark_counts": {str(mark): sum(event.mark == mark for event in events) for mark in range(3)},
        "same_customer_timestamp_extra_events": sum(max(0, count - 1) for count in timestamp_counts.values()),
        "aggregation_rule": "customer_id + invoice + invoice_date",
        "mark_rule": "0=cancellation; 1=purchase_abs_quantity<=threshold; 2=larger_purchase",
        "timestamp_timezone": "source-naive; no timezone conversion",
    }
    return events, profile


def load_workbook_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read and concatenate every non-empty workbook sheet with bundled pandas."""

    import pandas as pd

    sheets = pd.read_excel(Path(path), sheet_name=None)
    rows: list[dict[str, Any]] = []
    names: list[str] = []
    for name, frame in sheets.items():
        if frame.empty:
            continue
        names.append(str(name))
        rows.extend(frame.to_dict(orient="records"))
    if not rows:
        raise ValueError("workbook contains no data rows")
    return rows, names


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_processed(events: list[RetailEvent], profile: dict[str, Any],
                   *, npz_path: Path, profile_path: Path) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    customer_ids = np.asarray([event.customer_id for event in events])
    _, trajectory_index = np.unique(customer_ids, return_inverse=True)
    time_of_day_minutes = np.asarray(
        [event.timestamp.hour * 60 + event.timestamp.minute for event in events],
        dtype=np.int16,
    )
    np.savez_compressed(
        npz_path,
        trajectory_index=trajectory_index.astype(np.int32),
        time_of_day_minutes=time_of_day_minutes,
        mark=np.asarray([event.mark for event in events], dtype=np.int8),
    )
    profile.update(
        {
            "processed_schema": ["trajectory_index", "time_of_day_minutes", "mark"],
            "identifier_policy": "source customer and invoice identifiers are not retained",
            "timing_policy": "calendar dates are removed; only minute of day is retained",
        }
    )
    profile_path.write_text(json.dumps(profile, indent=2, allow_nan=False), encoding="utf-8")


def events_as_dicts(events: list[RetailEvent]) -> list[dict[str, Any]]:
    return [asdict(event) for event in events]

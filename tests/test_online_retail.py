from datetime import datetime

import numpy as np

from crome_identification.benchmarks.online_retail import (
    REQUIRED_COLUMNS,
    preprocess_rows,
    save_processed,
)


def _row(invoice, customer, when, quantity, stock="S1", price=2.0):
    return {
        "Invoice": invoice,
        "StockCode": stock,
        "Description": "item",
        "Quantity": quantity,
        "InvoiceDate": when,
        "Price": price,
        "Customer ID": customer,
        "Country": "United Kingdom",
    }


def test_required_columns_accept_both_uci_year_schemas():
    assert {"invoice", "quantity", "invoice_date", "customer_id"} <= set(REQUIRED_COLUMNS)


def test_preprocessing_filters_missing_aggregates_invoices_and_freezes_marks():
    when = datetime(2010, 1, 2, 10, 30)
    rows = [
        _row("100", 7, when, 8, "A"),
        _row("100", 7, when, 15, "B"),
        _row("C101", 7, datetime(2010, 1, 3, 9), -2),
        _row("102", None, datetime(2010, 1, 4, 9), 5),
    ]
    events, profile = preprocess_rows(rows, quantity_threshold=20, min_customer_events=1)
    assert len(events) == 2
    assert events[0].abs_quantity == 23
    assert events[0].mark == 2
    assert events[1].is_cancellation
    assert events[1].mark == 0
    assert profile["missing_customer_rows"] == 1
    assert profile["aggregation_rule"] == "customer_id + invoice + invoice_date"


def test_duplicate_rows_are_reported_but_not_double_counted():
    row = _row("200", 9, datetime(2010, 2, 1, 12), 4)
    events, profile = preprocess_rows([row, dict(row)], quantity_threshold=20, min_customer_events=1)
    assert len(events) == 1
    assert events[0].abs_quantity == 4
    assert profile["exact_duplicate_rows"] == 1


def test_customer_filter_and_order_are_deterministic():
    rows = [
        _row("1", 2, datetime(2010, 1, 1), 1),
        _row("2", 2, datetime(2010, 1, 2), 1),
        _row("3", 1, datetime(2010, 1, 3), 1),
    ]
    first, profile = preprocess_rows(rows, quantity_threshold=20, min_customer_events=2)
    second, _ = preprocess_rows(list(reversed(rows)), quantity_threshold=20, min_customer_events=2)
    assert first == second
    assert {event.customer_id for event in first} == {"2"}
    assert profile["eligible_customers"] == 1


def test_released_processed_file_omits_direct_transaction_identifiers(tmp_path):
    events, profile = preprocess_rows(
        [
            _row("100", 7, datetime(2010, 1, 2, 10, 30), 8),
            _row("101", 7, datetime(2010, 1, 3, 10, 30), 25),
        ],
        quantity_threshold=20,
        min_customer_events=1,
    )
    npz_path = tmp_path / "events.npz"
    profile_path = tmp_path / "profile.json"

    save_processed(events, profile, npz_path=npz_path, profile_path=profile_path)

    with np.load(npz_path, allow_pickle=False) as arrays:
        assert set(arrays.files) == {"customer_index", "timestamp_ns", "mark"}
        assert arrays["customer_index"].dtype.kind in {"i", "u"}

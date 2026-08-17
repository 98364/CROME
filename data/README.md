# Data

The repository includes the anonymized, minimized event-timing fixture used by
CS04 and its provenance profile. The preparation script retrieves the source
archive and workbook directly from UCI when rebuilding the fixture.

- Source: UCI Machine Learning Repository, Online Retail II
- Dataset DOI: https://doi.org/10.24432/C5CG6D
- Data license: CC BY 4.0
- Benchmark use: real transaction timing and quantity/cancellation-derived
  marks with controlled coarsening and synthetic responses; original prices
  and quantities are not used as benchmark responses.

`processed/online_retail_ii_profile.json` records the source archive, workbook,
and processed-file SHA-256 values. To download the UCI source and rebuild the
processed fixture locally:

```bash
python scripts/prepare_online_retail_ii.py --root .
```

The downloaded source files are written under `data/raw/`.

The checked-in NPZ contains only a contiguous customer-group index, the
source-naive event timestamp, and the frozen event mark. Source customer and
invoice identifiers, raw quantities, prices, cancellation flags, and product
fields are not retained.

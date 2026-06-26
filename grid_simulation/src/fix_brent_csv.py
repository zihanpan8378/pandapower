#!/usr/bin/env python3
"""Clean up Europe_Brent_Spot_Price_FOB.csv.

Fixes two problems:
  1. Inconsistent date strings missing leading zeros (e.g. "08/1/2020").
     Every date is normalized to zero-padded MM/DD/YYYY.
  2. Missing days. A row is inserted for *every* calendar day between the
     first and last date, and the price for each inserted day is
     forward-filled (carried over from the most recent known price).

The output keeps the original column header and newest-first ordering.
"""

import csv
import datetime as dt

INPUT = "Europe_Brent_Spot_Price_FOB.csv"
OUTPUT = "Europe_Brent_Spot_Price_FOB_fixed.csv"
DATE_FMT = "%m/%d/%Y"


def main() -> None:
    with open(INPUT, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Parse each row, tolerating missing leading zeros via %m/%d/%Y.
        records = {}
        for row in reader:
            if not row or not row[0].strip():
                continue
            date_str, price = row[0].strip(), row[1].strip()
            date = dt.datetime.strptime(date_str, DATE_FMT).date()
            records[date] = price  # dedupe: last occurrence wins

    if not records:
        raise SystemExit("No data rows found.")

    start, end = min(records), max(records)
    total_days = (end - start).days + 1

    # Walk every calendar day from oldest to newest, forward-filling prices.
    filled = []  # (date, price) ascending
    last_price = None
    inserted = 0
    for offset in range(total_days):
        day = start + dt.timedelta(days=offset)
        if day in records:
            last_price = records[day]
        else:
            inserted += 1
        # last_price is guaranteed set after the first iteration (day == start).
        filled.append((day, last_price))

    # Write out newest-first to match the original file's ordering.
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for day, price in reversed(filled):
            writer.writerow([day.strftime(DATE_FMT), price])

    print(f"Read {len(records)} dated rows ({start} -> {end}).")
    print(f"Wrote {len(filled)} rows to {OUTPUT} "
          f"({inserted} missing days inserted & forward-filled).")


if __name__ == "__main__":
    main()

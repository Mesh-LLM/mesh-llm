"""CSV processor — reads a CSV, filters rows, writes output."""
import csv
import sys
from datetime import datetime

def process_csv(input_path, output_path, min_date=None, max_amount=None):
    """Filter CSV rows by date and amount, write to output."""
    rows = []
    with open(input_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse date
            date = datetime.strptime(row['date'], '%Y-%m-%d')

            # Filter by min_date
            if min_date and date < min_date:
                continue

            # Filter by max_amount
            if max_amount and float(row['amount']) > max_amount:
                continue

            # Bug: modifying row dict while iterating
            row['date_parsed'] = date
            row['amount_float'] = float(row['amount'])
            rows.append(row)

    # Sort by amount descending
    rows.sort(key=lambda r: r['amount_float'])  # Fixed: sort as float

    # Write output
    if rows:
        with open(output_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            # Remove parsed fields before writing (datetime not serializable)
            rows_to_write = [{k: v for k, v in r.items() if k not in ('date_parsed', 'amount_float')} for r in rows]
            writer.writerows(rows_to_write)
            print(f"Wrote {len(rows)} rows")
    else:
        print("No rows matched filters")

    return len(rows)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python buggy.py input.csv output.csv [min_date] [max_amount]")
        sys.exit(1)

    min_date = None
    max_amount = None
    if len(sys.argv) > 3:
        min_date = datetime.strptime(sys.argv[3], '%Y-%m-%d')
    if len(sys.argv) > 4:
        max_amount = float(sys.argv[4])

    count = process_csv(sys.argv[1], sys.argv[2], min_date, max_amount)
    print(f"Processed {count} rows")

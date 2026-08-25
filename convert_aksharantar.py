"""
Convert Aksharantar Hindi CSVs to Dakshina-format TSVs.

Usage:
    python convert_aksharantar.py [--zip aksharantar_sampled.zip] [--outdir data]

Produces:
    data/hi.translit.sampled.train.tsv
    data/hi.translit.sampled.dev.tsv
    data/hi.translit.sampled.test.tsv

Each output line:  <native>\t<roman>\t1
"""

import argparse
import csv
import io
import os
import zipfile


SPLIT_MAP = {
    "train": "hin_train.csv",
    "dev": "hin_valid.csv",
    "test": "hin_test.csv",
}


def read_csv_from_zip(zip_path, inner_name):
    with zipfile.ZipFile(zip_path, "r") as z:
        # Find the hin CSV regardless of top-level folder name
        matching = [n for n in z.namelist() if n.endswith(inner_name)]
        if not matching:
            raise FileNotFoundError(f"{inner_name} not found in {zip_path}")
        with z.open(matching[0]) as f:
            text = io.TextIOWrapper(f, encoding="utf-8")
            reader = csv.reader(text)
            pairs = []
            for row in reader:
                if len(row) >= 2:
                    roman = row[0].strip()
                    native = row[1].strip()
                    if roman and native:
                        pairs.append((native, roman))
    return pairs


def write_tsv(pairs, outpath):
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    with open(outpath, "w", encoding="utf-8", newline="\n") as f:
        for native, roman in pairs:
            f.write(f"{native}\t{roman}\t1\n")


def main():
    parser = argparse.ArgumentParser(description="Convert Aksharantar to Dakshina TSV format")
    parser.add_argument("--zip", default="aksharantar_sampled.zip", help="Path to aksharantar zip")
    parser.add_argument("--outdir", default="data", help="Output directory for TSVs")
    args = parser.parse_args()

    if not os.path.exists(args.zip):
        raise SystemExit(f"Zip file not found: {args.zip}")

    for split, fname in SPLIT_MAP.items():
        pairs = read_csv_from_zip(args.zip, fname)
        outpath = os.path.join(args.outdir, f"hi.translit.sampled.{split}.tsv")
        write_tsv(pairs, outpath)
        print(f"Wrote {len(pairs):>6} pairs -> {outpath}")

    print("Done. Set DAKSHINA_DATA_DIR to the output directory.")


if __name__ == "__main__":
    main()

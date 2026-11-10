import csv
from pathlib import Path

def test_synthetic_dataset_has_five_balanced_classes():
    root=Path(__file__).resolve().parents[2]/"dataset"/"synthetic"
    rows=list(csv.DictReader((root/"manifest.csv").open(encoding="utf-8")))
    assert len(rows)==60
    counts={}
    for row in rows: counts[row["document_type"]]=counts.get(row["document_type"],0)+1
    assert len(counts)==5
    assert set(counts.values())=={12}

def test_each_manifest_image_and_transcript_exists():
    root=Path(__file__).resolve().parents[2]/"dataset"/"synthetic"
    rows=list(csv.DictReader((root/"manifest.csv").open(encoding="utf-8")))
    assert all((root/r["image_path"]).exists() for r in rows)
    assert all((root/r["transcript_path"]).exists() for r in rows)

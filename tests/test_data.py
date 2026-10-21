import zipfile
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from perfume_classifier import checks
from perfume_classifier.data import load_catalog, parse_ingredients, safe_extract

HEADER = "brand|name_perfume|family|subfamily|fragrances|ingredients|origin|gender|years|description|image_name\n"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("['Rose', 'Musk']", ["Rose", "Musk"]),
        ("[' Rose ', '']", ["Rose"]),
        ("[]", []),
        ("", []),
        (None, []),
        ("not a list", []),
        ("{'a': 1}", []),
        ("['unclosed", []),
    ],
)
def test_parse_ingredients(raw: object, expected: list[str]) -> None:
    assert parse_ingredients(raw) == expected


def test_load_catalog_turns_blanks_into_na(tmp_path: Path) -> None:
    csv = tmp_path / "c.csv"
    csv.write_text(HEADER + "Dior|Sauvage|WOODY|WOODY|Woody|['Pepper']| |Male|2015||a.jpg\n", encoding="utf-8")
    df = load_catalog(csv)
    assert df.loc[0, "ingredients"] == ["Pepper"]
    assert pd.isna(df.loc[0, "origin"])
    assert pd.isna(df.loc[0, "description"])
    assert df.loc[0, "brand"] == "Dior"


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with pytest.raises(ValueError):
        safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_check_image_flags_broken_and_accepts_good(tmp_path: Path) -> None:
    good = tmp_path / "good.jpg"
    Image.new("RGB", (8, 8), "red").save(good)
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(good.read_bytes()[:20])
    empty = tmp_path / "empty.jpg"
    empty.write_bytes(b"")
    assert checks.check_image(good) == (True, (8, 8), "RGB")
    assert checks.check_image(broken)[0] is False
    assert checks.check_image(empty)[0] is False
    assert checks.check_image(tmp_path / "missing.jpg")[0] is False


def test_duplicates_ignore_missing_image_names() -> None:
    df = pd.DataFrame({
        "brand": ["A", "a", "B"],
        "name_perfume": ["X", "x", "Y"],
        "image_name": [pd.NA, pd.NA, "1.jpg"],
    })
    dups = checks.duplicate_counts(df)
    assert dups["image_name"] == 0
    assert dups["brand_and_name"] == 0
    assert dups["brand_and_name_case_insensitive"] == 1


def test_usable_rows(tmp_path: Path) -> None:
    Image.new("RGB", (4, 4)).save(tmp_path / "ok.jpg")
    Image.new("RGB", (4, 4)).save(tmp_path / "bad.jpg")
    df = pd.DataFrame({"image_name": ["ok.jpg", "bad.jpg", "gone.jpg", pd.NA]})
    assert checks.usable_rows(df, tmp_path, ["bad.jpg"]).tolist() == [True, False, False, False]

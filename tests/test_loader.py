from ml_starter.data.loader import load_uci_cleveland


def test_loader_basic(tmp_path):
    # สร้างไฟล์ตัวอย่าง
    p = tmp_path / "cleveland.csv"
    p.write_text("63,1,3,145,233,1,0,150,0,2.3,0,?,6,0\n")
    df = load_uci_cleveland(str(p))
    assert "target" in df.columns
    assert df["target"].iloc[0] in (0, 1)
    assert df.isna().sum().sum() >= 1  # "?" -> NaN

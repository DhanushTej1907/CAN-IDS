import pandas as pd
import pytest

from src.load import BYTE_COLS, clean_chunk, load_processed, parse_file_meta, process_file

META = {"mode": "S", "subset": "prelim_train", "source_file": "Pre_train_S_9"}


def raw(rows, with_subclass=True):
    cols = ["Timestamp", "Arbitration_ID", "DLC", "Data", "Class"] + (["SubClass"] if with_subclass else [])
    df = pd.DataFrame(rows, columns=cols)
    return df.astype({"Timestamp": "float64", "DLC": "int16"})


def test_parse_file_meta():
    assert parse_file_meta("data/raw/0_Preliminary/0_Training/Pre_train_D_1.csv") == {
        "mode": "D", "subset": "prelim_train", "source_file": "Pre_train_D_1"}
    assert parse_file_meta("x/1_Submission/Pre_submit_S.csv")["subset"] == "prelim_submit"
    assert parse_file_meta("x/1_Final/Fin_host_session_submit_S.csv") == {
        "mode": "S", "subset": "final", "source_file": "Fin_host_session_submit_S"}
    with pytest.raises(ValueError):
        parse_file_meta("x/0_Training/no_mode_here.csv")


def test_hex_parsing_and_padding():
    df = clean_chunk(raw([
        [1597707827.052591, "260", 8, "06 25 05 30 FF CF 71 55", "Normal", "Normal"],
        [1597707827.05467, "7FF", 4, "08 00 00 01", "Attack", "Flooding"],
        [1597707827.06, "0", 0, "", "Normal", "Normal"],
    ]), META, row_offset=10)
    assert df["arb_id"].tolist() == [0x260, 0x7FF, 0]
    assert df[BYTE_COLS].iloc[0].tolist() == [0x06, 0x25, 0x05, 0x30, 0xFF, 0xCF, 0x71, 0x55]
    assert df[BYTE_COLS].iloc[1].tolist() == [8, 0, 0, 1, -1, -1, -1, -1]
    assert df[BYTE_COLS].iloc[2].tolist() == [-1] * 8
    assert df["n_bytes"].tolist() == [8, 4, 0]
    assert df["row_idx"].tolist() == [10, 11, 12]
    assert df["is_attack"].tolist() == [0, 1, 0]
    assert df["timestamp"].iloc[0] == 1597707827.052591
    assert not df["ts_relative"].any()


def test_missing_subclass_filled_for_normal_only_file():
    df = clean_chunk(raw([[1.6e9, "130", 2, "AB CD", "Normal"]], with_subclass=False), META, 0)
    assert df["subclass"].tolist() == ["Normal"]


def test_missing_subclass_with_attack_raises():
    with pytest.raises(ValueError, match="without SubClass"):
        clean_chunk(raw([[1.6e9, "130", 2, "AB CD", "Attack"]], with_subclass=False), META, 0)


@pytest.mark.parametrize("row, msg", [
    ([1.6e9, "XYZ", 1, "00", "Normal", "Normal"], "Arbitration_ID"),
    ([1.6e9, "800", 1, "00", "Normal", "Normal"], "Arbitration_ID"),
    ([1.6e9, "130", 1, "G0", "Normal", "Normal"], "payload byte 0"),
    ([1.6e9, "130", 9, "00", "Normal", "Normal"], "DLC"),
    ([1.6e9, "130", 1, "00", "Normal", "DoS"], "SubClass"),
])
def test_strict_validation_raises_with_row(row, msg):
    with pytest.raises(ValueError, match=msg):
        clean_chunk(raw([row]), META, row_offset=41)


def test_process_file_roundtrip(tmp_path):
    src = tmp_path / "raw" / "0_Training" / "Pre_train_S_9.csv"
    src.parent.mkdir(parents=True)
    src.write_text(
        "Timestamp,Arbitration_ID,DLC,Data,Class,SubClass\n"
        "16.05212,58B,8,11 00 00 00 00 00 00 00,Normal,Normal\n"
        "16.05236,130,0,,Attack,Fuzzing\n"
        "16.0526,140,2,80 7F,Attack,Replay\n"
    )
    out = tmp_path / "processed"
    r = process_file(src, out, chunksize=2)  # 2 chunks -> exercises multi-row-group writing
    assert r["rows"] == 3 and r["attack_rows"] == 2

    df = load_processed(subset="prelim_train", processed_dir=out)
    assert len(df) == 3
    assert df["ts_relative"].all()
    assert df["subclass"].tolist() == ["Normal", "Fuzzing", "Replay"]
    assert df["byte_1"].tolist() == [0, -1, 0x7F]
    with pytest.raises(FileNotFoundError):
        load_processed(subset="final", processed_dir=out)

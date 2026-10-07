from src.clean_checks import check_file
from src.load import process_file


def _statuses(tmp_path, body):
    src = tmp_path / "raw" / "0_Training" / "Pre_train_S_9.csv"
    src.parent.mkdir(parents=True)
    src.write_text("Timestamp,Arbitration_ID,DLC,Data,Class,SubClass\n" + body)
    process_file(src, tmp_path / "processed")
    r = check_file(tmp_path / "processed" / "Pre_train_S_9.parquet")
    return {c["check"]: (c["status"], c["count"]) for c in r["checks"]}


def test_clean_file_all_ok(tmp_path):
    s = _statuses(tmp_path, "1.0,130,2,AA BB,Normal,Normal\n1.1,140,1,CC,Normal,Normal\n")
    assert all(st == "OK" for st, _ in s.values())


def test_detects_planted_problems(tmp_path):
    s = _statuses(tmp_path,
                  "1.0,130,2,AA BB,Normal,Normal\n"
                  "1.0,130,2,AA BB,Normal,Normal\n"     # exact dup of Normal row -> FLAG
                  "0.9,140,3,CC,Normal,Normal\n"        # goes backwards; DLC 3 vs 1 byte
                  "2.0,150,1,DD,Attack,Replay\n"
                  "2.0,150,1,DD,Attack,Replay\n"        # also duplicate (Attack)
                  "3.0,160,1,EE,Attack,Normal\n")       # inconsistent labels
    assert s["exact_duplicate_rows"] == ("FLAG", 2)
    assert s["dlc_vs_byte_count_mismatch"] == ("FLAG", 1)
    assert s["timestamp_decreasing"] == ("FLAG", 1)
    assert s["timestamp_ties"] == ("INFO", 2)
    assert s["class_subclass_inconsistent"] == ("FLAG", 1)


def test_attack_only_duplicates_are_info(tmp_path):
    s = _statuses(tmp_path, "1.0,130,1,AA,Attack,Replay\n1.0,130,1,AA,Attack,Replay\n")
    assert s["exact_duplicate_rows"] == ("INFO", 1)

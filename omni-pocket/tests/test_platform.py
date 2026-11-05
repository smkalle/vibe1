from omnipocket.platform import detect_env, mem_info, recommend


def test_detect_termux():
    assert detect_env({"PREFIX": "/data/data/com.termux/files/usr"}, proc_root="/nonexistent") == "termux"


def test_detect_proot_by_env():
    assert detect_env({"PROOT_TMP_DIR": "/tmp"}, proc_root="/nonexistent") == "proot"


def test_detect_proot_by_tracer(tmp_path):
    (tmp_path / "self").mkdir()
    (tmp_path / "self" / "status").write_text("Name:\tpython\nTracerPid:\t42\n")
    (tmp_path / "42").mkdir()
    (tmp_path / "42" / "comm").write_text("proot\n")
    assert detect_env({}, proc_root=str(tmp_path)) == "proot"


def test_mem_info(tmp_path):
    p = tmp_path / "meminfo"
    p.write_text("MemTotal:       11811160 kB\nMemFree: 1 kB\nMemAvailable:    4194304 kB\n")
    assert mem_info(str(p)) == {"total_mb": 11534, "available_mb": 4096}
    assert mem_info(str(tmp_path / "missing")) == {"total_mb": 0, "available_mb": 0}


def test_recommend_ram_fit():
    notes = recommend("termux", 8 * 1024, {"streamlit": True})
    text = "\n".join(notes)
    assert "PyTorch wheels are not on PyPI" in text
    assert "bfloat16: needs ~4.4 GB for weights -> fits in 8.0 GB RAM" in text
    assert "float32: needs ~8.7 GB for weights -> too big" in text

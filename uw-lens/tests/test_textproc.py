import pytest

from uwlens.textproc import chunk_text, detect_language, find_phi, redact_phi, tokenize


def test_tokenize_latin_and_cjk_bigrams():
    toks = tokenize("Sleep apnea and 高額療養費")
    assert "sleep" in toks and "apnea" in toks and "and" not in toks
    assert {"高額", "額療", "療養", "養費"} <= set(toks)


def test_tokenize_folds_fullwidth():
    assert tokenize("ＨｂＡ１ｃ") == tokenize("HbA1c") == ["hba1c"]


@pytest.mark.parametrize("text,lang", [
    ("Type 2 diabetes is a chronic condition.", "en"),
    ("高額療養費制度は、医療費の自己負担が高額になった場合の制度です。", "ja"),
    ("HbA1c 6.9 % 判定 要医療 治療中 BMI 29.4 腹囲 92 cm metabolic", "mixed"),
    ("12345", "en"),
])
def test_detect_language(text, lang):
    assert detect_language(text) == lang


def test_chunk_text_bounds_and_coverage():
    text = " ".join(f"Sentence number {i} about HbA1c and blood pressure." for i in range(200))
    chunks = chunk_text(text, max_chars=400, overlap=80)
    assert len(chunks) > 5
    assert all(len(c) <= 400 for c in chunks)
    for i in range(200):  # every sentence survives chunking
        assert any(f"Sentence number {i} " in c for c in chunks)
    # overlap snaps to whole sentences: chunks never start mid-word
    assert all(c.startswith("Sentence") for c in chunks)


def test_chunk_text_hard_splits_long_lines_and_japanese():
    ja = "高額療養費" * 500  # no sentence terminators at all
    chunks = chunk_text(ja, max_chars=300, overlap=50)
    assert all(len(c) <= 300 for c in chunks)
    assert "".join(chunks).count("高額療養費") >= 500
    assert chunk_text("", 100) == [] and chunk_text("short", 100) == ["short"]


def test_redact_phi_english_and_japanese_keeps_clinical_values():
    raw = (
        "Applicant ID: UW-2026-00417\nPatient name: Taro Example\nDOB: 1971-03-14\n"
        "Phone: 090-1234-5678, (415) 555-0199 · Email: taro@example.jp · SSN 123-45-6789 · "
        "My Number 1234 5678 9012\n被保険者番号：48271936　氏名：例　太郎\n生年月日：昭和46年3月14日\n"
        "Policy No. TL-889-221. HbA1c 7.2% on 2026-05-01, BP 142/91 mmHg, eGFR 71, AHI 22.4."
    )
    red, counts = redact_phi(raw)
    for secret in ("UW-2026-00417", "Taro Example", "1971-03-14", "090-1234-5678", "555-0199", "taro@example.jp",
                   "123-45-6789", "1234 5678 9012", "48271936", "例", "昭和46年", "TL-889-221"):
        assert secret not in red, secret
    for keep in ("HbA1c 7.2%", "2026-05-01", "142/91 mmHg", "eGFR 71", "AHI 22.4"):
        assert keep in red, keep
    assert {"MEMBER_ID", "NAME", "DOB", "PHONE", "EMAIL", "SSN", "MY_NUMBER"} <= set(counts)
    assert find_phi(red) == []


def test_redaction_is_idempotent():
    once, _ = redact_phi("Member ID: ABC12345 call 03-1234-5678")
    twice, counts = redact_phi(once)
    assert once == twice and counts == {}

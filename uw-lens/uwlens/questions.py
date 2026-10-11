"""Sample underwriting research questions, grouped by category, for the Search tab drop-downs.

Each question names the sample-corpus file(s) that should answer it; tests check they rank in the
top 5, so the samples stay useful as the corpus or ranking changes.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SampleQuestion:
    text: str
    expect: tuple[str, ...]  # sample-corpus file names that answer it


CATEGORIES: dict[str, tuple[SampleQuestion, ...]] = {
    "🩸 Diabetes & metabolic": (
        SampleQuestion("What HbA1c level confirms a diabetes diagnosis?", ("type2_diabetes.md",)),
        SampleQuestion("How is type 2 diabetes rated when HbA1c is above 8%?", ("underwriting_manual.pdf",)),
        SampleQuestion("メタボリックシンドロームの判定基準は？", ("tokutei_kenshin.md",)),
    ),
    "😴 Sleep apnea": (
        SampleQuestion("How is sleep apnea severity graded by AHI?", ("obstructive_sleep_apnea.md",)),
        SampleQuestion("What counts as good CPAP compliance for underwriting?",
                       ("underwriting_manual.pdf", "obstructive_sleep_apnea.md", "rating_rules.py")),
        SampleQuestion("What did the applicant's home sleep test show?", ("sleep_study_0417.png",)),
    ),
    "❤️ Cardiovascular & kidney": (
        SampleQuestion("What are the stage 1 and stage 2 hypertension thresholds?", ("hypertension.md",)),
        SampleQuestion("How does CHA2DS2-VASc affect the atrial fibrillation rating?",
                       ("underwriting_manual.pdf", "atrial_fibrillation.md")),
        SampleQuestion("What eGFR and albuminuria stages define chronic kidney disease?",
                       ("chronic_kidney_disease.md",)),
    ),
    "🇯🇵 Japan public insurance": (
        SampleQuestion("高額療養費の自己負担限度額はいくらですか？", ("kogaku_ryoyohi.md",)),
        SampleQuestion("介護保険の第2号被保険者が使える特定疾病とは？", ("kaigo_hoken.md",)),
        SampleQuestion("What co-payment rates apply under Japan's public health insurance?",
                       ("nhi_fee_schedule_en.md",)),
    ),
    "📜 Disclosure & underwriting rules": (
        SampleQuestion("告知義務違反があった場合、契約は解除されますか？", ("kokuchi_gimu.pdf",)),
        SampleQuestion("How many BMI debits apply to an obese applicant?", ("underwriting_manual.pdf", "rating_rules.py")),
        SampleQuestion("Can a Japanese health checkup replace the paramedical exam?",
                       ("underwriting_manual.pdf", "nhi_fee_schedule_en.md")),
    ),
    "🗂️ Case 0417 evidence": (
        SampleQuestion("What medications is the applicant taking?", ("aps_case_0417.md", "teleinterview_0417.wav")),
        SampleQuestion("Has the applicant's HbA1c improved over time?", ("bp_trend_0417.png", "aps_case_0417.md")),
        SampleQuestion("Does the applicant use CPAP every night?",
                       ("cpap_compliance_0417.mp4", "teleinterview_0417.wav", "aps_case_0417.md")),
    ),
}

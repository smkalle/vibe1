# UW Lens retrieval eval

- Run: 2026-10-11 09:29 UTC
- Embedder: Offline hashing embedder (lexical fallback) (`hashing-v1-768`)
- Corpus: synthetic sample, 19 files / 28 chunks, skipped 0
- Queries: 25 (`evals/queries.json`); a hit = any expected file in top-5 (grouped by file)

Cross-lingual queries (EN query → JA doc and vice versa) need a semantic multilingual model; the
lexical hashing fallback is expected to miss them. They are reported separately and excluded from
the same-language gate.

| Fusion | Group | n | Recall@5 | MRR |
|---|---|---|---|---|
| keyword | cross_lingual | 4 | 0.250 | 0.250 |
| keyword | en | 16 | 1.000 | 0.875 |
| keyword | ja | 5 | 1.000 | 1.000 |
| keyword | all_same_language | 21 | 1.000 | 0.905 |
| semantic | cross_lingual | 4 | 0.500 | 0.333 |
| semantic | en | 16 | 1.000 | 0.938 |
| semantic | ja | 5 | 1.000 | 1.000 |
| semantic | all_same_language | 21 | 1.000 | 0.952 |
| rrf | cross_lingual | 4 | 0.500 | 0.312 |
| rrf | en | 16 | 1.000 | 0.938 |
| rrf | ja | 5 | 1.000 | 1.000 |
| rrf | all_same_language | 21 | 1.000 | 0.952 |
| weighted | cross_lingual | 4 | 0.500 | 0.333 |
| weighted | en | 16 | 1.000 | 0.906 |
| weighted | ja | 5 | 1.000 | 1.000 |
| weighted | all_same_language | 21 | 1.000 | 0.929 |

## Per-query ranks (rrf)

| Query | Group | Rank | Top-3 |
|---|---|---|---|
| CPAP compliance 4 hours 70% of nights | en | 1 | cpap_compliance_0417.mp4, aps_case_0417.md, rating_rules.py |
| how is sleep apnea severity graded by AHI | en | 2 | sleep_study_0417.png, obstructive_sleep_apnea.md, underwriting_manual.pdf |
| applicant sleep study lowest oxygen saturation | en | 2 | obstructive_sleep_apnea.md, sleep_study_0417.png, underwriting_manual.pdf |
| HbA1c above 8 percent rating table diabetes | en | 1 | underwriting_manual.pdf, rating_rules.py, type2_diabetes.md |
| diagnosis criteria fasting plasma glucose 126 | en | 1 | type2_diabetes.md, hypertension.md, chronic_kidney_disease.md |
| stage 2 hypertension blood pressure categories | en | 1 | hypertension.md, obstructive_sleep_apnea.md, bp_trend_0417.png |
| left ventricular hypertrophy target organ damage | en | 1 | underwriting_manual.pdf, hypertension.md, chronic_kidney_disease.md |
| eGFR stages and albuminuria categories | en | 1 | chronic_kidney_disease.md, type2_diabetes.md, underwriting_manual.pdf |
| CHA2DS2-VASc stroke risk anticoagulation | en | 1 | atrial_fibrillation.md, underwriting_manual.pdf, type2_diabetes.md |
| asthma exacerbation intubation risk | en | 1 | asthma.md, atrial_fibrillation.md, type2_diabetes.md |
| BMI build debits obesity | en | 1 | underwriting_manual.pdf, rating_rules.py, atrial_fibrillation.md |
| function computing body mass index debits | en | 1 | rating_rules.py, underwriting_manual.pdf, asthma.md |
| applicant medications metformin empagliflozin telmisartan | en | 1 | teleinterview_0417.wav, aps_case_0417.md, type2_diabetes.md |
| tele-interview smoking history | en | 1 | teleinterview_0417.wav, nhi_fee_schedule_en.md, hypertension.md |
| trend chart of HbA1c over time | en | 1 | bp_trend_0417.png, type2_diabetes.md, kenshin_result_0417.png |
| Japan co-payment rate 30% fee schedule points 10 yen | en | 1 | nhi_fee_schedule_en.md, atrial_fibrillation.md, underwriting_manual.pdf |
| 高額療養費 自己負担限度額 多数回該当 | ja | 1 | kogaku_ryoyohi.md, kaigo_hoken.md, kenshin_result_0417.png |
| メタボリックシンドロームの判定基準 腹囲 | ja | 1 | tokutei_kenshin.md, kenshin_result_0417.png, kaigo_hoken.md |
| 要介護認定 第2号被保険者 特定疾病 | ja | 1 | kaigo_hoken.md, kokuchi_gimu.pdf, tokutei_kenshin.md |
| 告知義務違反 契約解除 5年 | ja | 1 | kokuchi_gimu.pdf, aps_case_0417.md, kaigo_hoken.md |
| 健康診断結果 HbA1c 尿糖 | ja | 1 | kenshin_result_0417.png, kokuchi_gimu.pdf, tokutei_kenshin.md |
| monthly out-of-pocket cap for high medical costs in Japan | cross_lingual | 1 | nhi_fee_schedule_en.md, type2_diabetes.md, hypertension.md |
| long-term care insurance certification levels | cross_lingual | — | nhi_fee_schedule_en.md, asthma.md, hypertension.md |
| duty of disclosure when applying for life insurance | cross_lingual | — | nhi_fee_schedule_en.md, underwriting_manual.pdf, hypertension.md |
| 睡眠時無呼吸症候群の重症度 | cross_lingual | 4 | kokuchi_gimu.pdf, tokutei_kenshin.md, kogaku_ryoyohi.md |

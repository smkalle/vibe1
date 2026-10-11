"""Debit-point helpers for the FICTIONAL Example Mutual underwriting manual (UW Lens demo corpus).

Illustrative code only: not a real rating engine.
"""


def diabetes_table(hba1c: float, insulin: bool = False, complications: bool = False) -> str:
    """Map latest HbA1c (percent) to a substandard table per manual section 3."""
    if hba1c > 10.0 or complications:
        return "Decline"
    if hba1c > 8.0 or insulin:
        return "Table 4-6"
    if hba1c >= 7.0:
        return "Table 2-4"
    return "Standard-Table 2"


def bmi_debits(weight_kg: float, height_cm: float) -> int:
    """Build debits from body-mass index per manual section 2."""
    bmi = weight_kg / (height_cm / 100) ** 2
    if bmi >= 40:
        return 100  # Table 4 minimum
    if bmi >= 35:
        return 50
    if bmi >= 30:
        return 25
    return 0


def cpap_compliant(nights_used_4h: int, nights_total: int) -> bool:
    """Sleep apnea CPAP adherence: 4+ hours on at least 70% of nights."""
    return nights_total > 0 and nights_used_4h / nights_total >= 0.70

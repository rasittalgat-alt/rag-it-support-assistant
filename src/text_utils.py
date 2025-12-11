# src/text_utils.py
from typing import Dict


COMMON_TYPO_MAP: Dict[str, str] = {
    # printers
    "preinter": "printer",
    "preinters": "printers",
    "prnter": "printer",
    "prnters": "printers",

    # wifi
    "wfi": "wifi",
    "wi fi": "wifi",

    # vpn
    "vnp": "vpn",

    # email
    "eamil": "email",
    "maill": "mail",

    # password
    "passwrod": "password",
    "pasword": "password",
}


def normalize_question(text: str) -> str:
    """
    Простейшая нормализация вопроса:
    - приведение к lower
    - замена частых опечаток доменных слов
    """
    q = text.lower()
    for typo, correct in COMMON_TYPO_MAP.items():
        q = q.replace(typo, correct)
    return q

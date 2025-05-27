# -*- coding: utf-8 -*-
import cv2
import pytesseract
import re
from PIL import Image
from fastai.vision.all import load_learner
import numpy as np
import pathlib, platform, shutil
from pathlib import Path
from typing import Tuple, Union, List
import pandas as pd

# === PLATFORM UCHUN TAYYORLASH ===
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath
    pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
elif shutil.which("tesseract") is None:
    raise EnvironmentError("Tesseract OCR topilmadi!")

# === MODELNI YUKLASH ===
try:
    base_path = Path(__file__).parent
except NameError:
    base_path = Path.cwd()

MODEL_PATH = (base_path / "Model_14_1_1.pkl").resolve()
if not MODEL_PATH.exists():
    raise FileNotFoundError("❌ Model fayli topilmadi: Model_14_1_1.pkl")

_learn = load_learner(MODEL_PATH)

# === SERTIFIKATNI ANIQLASH ===
def classify_certificate(img: Image.Image) -> Tuple[str, float]:
    pred, pred_id, probs = _learn.predict(img)
    return str(pred).lower(), float(probs[pred_id])

# === PIL → OpenCV konvertatsiya ===
def _pil2cv(im: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

# === Normalize qilingan sertifikat tiplari ===
def normalize_cert_type(cert_type: str) -> str:
    cert_type = cert_type.strip().lower().replace(" ", "")
    mapping = {
        "toefl": "TOEFL",
        "toefloverall": "TOEFL",
        "toeflibt": "TOEFL",
        "ielts": "IELTS",
        "duolingo": "DUOLINGO",
        "sat": "SAT",
        "gre": "GRE",
        "topik": "TOPIK"
    }
    return mapping.get(cert_type, cert_type.upper())

# === IELTS score extractor ===

def _extract_ielts(cv_img):
    height, width = cv_img.shape[:2]

    # Band score joylashgan qism
    y1 = int(height * 0.45)
    y2 = int(height * 0.60)
    x1 = int(width * 0.75)
    x2 = int(width * 0.80)

    white_bg = np.ones_like(cv_img, dtype=np.uint8) * 255

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    focused = cv2.bitwise_and(cv_img, cv_img, mask=mask)
    inverse_mask = cv2.bitwise_not(mask)
    white_only = cv2.bitwise_and(white_bg, white_bg, mask=inverse_mask)
    result = cv2.add(focused, white_only)

    cropped = result[y1:y2, x1:x2]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    config = r'--oem 3 --psm 7 outputbase digits'
    text = pytesseract.image_to_string(thresh, config=config)

    # ✅ Tozalash va nuqta o‘rnini to‘g‘rilash
    cleaned = ''.join(c for c in text if c.isdigit() or c == '.')
    
    # OCR 65 deb o‘qigan bo‘lsa, aslida 6.5 bo‘lishi mumkin (faqat 2 raqam bo‘lsa)
    if cleaned.isdigit() and len(cleaned) == 2:
        cleaned = cleaned[0] + "." + cleaned[1]

    return cleaned


# === TOEFL extractor ===
def _extract_toefl(cv_img):
    try:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh, config="--psm 6")
        match = re.search(r"\b([5-9][0-9]|1[01][0-9]|120)\b", text)
        if match:
            return int(match.group(1))
    except Exception:
        return None

# === DUOLINGO extractor ===

import cv2
import pytesseract
from difflib import get_close_matches

def _extract_overall_duolingo(cv_img):
    if cv_img is None or cv_img.size == 0:
        print("❌ Bo‘sh rasm yoki noto‘g‘ri rasm.")
        return None

    # Rasmni oq-qora va threshold qilish
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # OCR orqali matnlarni olish
    ocr_data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)

    results = []
    for i in range(len(ocr_data['text'])):
        text = ocr_data['text'][i].strip()
        if text:
            left = ocr_data['left'][i]
            top = ocr_data['top'][i]
            width = ocr_data['width'][i]
            height = ocr_data['height'][i]
            results.append((text, left, top, width, height))

    # 1. "Overall" ga o‘xshash so‘zni topamiz
    words = [t[0] for t in results]
    match = get_close_matches("Overall", words, n=1, cutoff=0.6)

    if not match:
        print("❌ 'Overall' so‘zi topilmadi (hatto o‘xshashi ham yo‘q).")
        return None

    matched_word = match[0]
    overall_coords = None
    for word, left, top, width, height in results:
        if word == matched_word:
            overall_coords = (left, top)
            break

    if overall_coords is None:
        print("❌ 'Overall' koordinatasi aniqlanmadi.")
        return None

    overall_x, overall_y = overall_coords

    # 2. Uning chapida joylashgan mos raqamlarni topamiz (80–160 oralig‘ida)
    possible_scores = []
    for word, left, top, _, _ in results:
        if word.isdigit() and 80 <= int(word) <= 160:
            if abs(top - overall_y) < 50 and left < overall_x:
                possible_scores.append((int(word), left))

    if not possible_scores:
        print("❌ Chap tarafda mos raqam topilmadi.")
        return None

    # Eng yaqin chapdagi raqam
    possible_scores.sort(key=lambda x: abs(x[1] - overall_x))
    score = possible_scores[0][0]

    print(f"✅ Duolingo Overall Score: {score}")
    return score



# === SAT extractor ===
def _extract_sat(cv_img):
    h, w = cv_img.shape[:2]
    roi = cv_img[int(0.35 * h):int(0.45 * h), :w // 3]
    txt = pytesseract.image_to_string(roi)
    m = re.search(r"Your Total Score\s*(\d{3,4})", txt, re.I)
    return int(m.group(1)) if m else None

# === TOPIK extractor ===
def _extract_topik(cv_img):
    h, w = cv_img.shape[:2]
    roi = cv_img[int(0.35 * h):int(0.62 * h), int(0.50 * w):int(0.88 * w)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    ocr_text = pytesseract.image_to_string(bw, config="--psm 6").strip()

    numbers = [int(n) for n in re.findall(r'\b\d{2,3}\b', ocr_text) if 80 <= int(n) <= 300]
    if not numbers:
        return None

    score = min(numbers)
    if 80 <= score <= 139:
        level = 1
    elif 140 <= score <= 149:
        level = 2
    elif 150 <= score <= 189:
        level = 3
    elif 190 <= score <= 229:
        level = 4
    elif 230 <= score <= 300:
        level = 5
    else:
        level = None

    return level

# === Extractorlar lug'ati ===
_EXTRACTORS = {
    "toefl": _extract_toefl,
    "duolingo": _extract_overall_duolingo,
    "ielts": _extract_ielts,
    "sat": _extract_sat,
    "topik": _extract_topik
}

# === Yakuniy score chiqaruvchi ===
def extract_score(cert_type: str, img: Image.Image) -> Union[int, float, None]:
    cert_type_lower = cert_type.lower()
    fn = _EXTRACTORS.get(cert_type_lower)
    if not fn:
        raise ValueError(f"Unknown cert type: {cert_type}")
    return fn(_pil2cv(img))

# === Universitetlarni sertifikat ballari asosida filtrlash ===
from typing import List, Tuple, Union
import pandas as pd

def filter_universities(df: pd.DataFrame, cert_scores: List[Tuple[str, Union[int, float], str]]) -> pd.DataFrame:
    column_map = {
        "TOEFL": "TOEFL iBT",
        "IELTS": "IELTS",
        "DUOLINGO": "Duolingo",
        "SAT": "SAT",
        "GRE": "GRE",
        "TOPIK": "TOPIK_level"
    }

    # Kolonkalardagi qiymatlarni floatga aylantirish
    for col in column_map.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filtrlovchi funksiya
    def match_all_requirements(row):
        for cert_type, user_score, _ in cert_scores:
            norm_type = cert_type.upper().strip()
            if norm_type not in column_map:
                return False

            col_name = column_map[norm_type]
            if col_name not in row or pd.isna(row[col_name]):
                return False

            required = row[col_name]

            try:
                if norm_type == "TOPIK":
                    if int(required) > int(user_score):
                        return False
                else:
                    # Foydalanuvchi ballini to'g'ridan floatga aylantirish (6.5, 7.0)
                    user_score_clean = float(user_score)
                    required_clean = float(required)
                    if required_clean > user_score_clean:
                        return False
            except Exception:
                return False
        return True

    # Har bir qatorga `match_all_requirements` ni qo‘llash
    return df[df.apply(match_all_requirements, axis=1)]




 
# BFT65B – Big-Data Pipeline for Bank Failure Early Warning (FDIC)

Author: <ten> – <MSSV>

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python src/01_download.py --probe       # kiem tra truong
python src/01_download.py               # tai toan bo (resumable, ~1 gio)
```
(Cac buoc tiep theo se duoc bo sung.)

## Data
FDIC BankFind API (https://api.fdic.gov/banks). Du lieu khong dua len Git.

## Citations & AI usage
- Data: FDIC BankFind Suite API.
- Starter script: `fdic_download_starter.py` (giang vien cung cap).
- AI tools: Claude (Anthropic) – ho tro viet script tai du lieu va len ke hoach. Ghi them cac phan khac khi dung.

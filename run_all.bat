@echo off
REM Chay toan bo pipeline tu dau den cuoi (Windows). Bam dup file nay hoac chay: run_all.bat
cd /d %~dp0

if not exist .venv (
    python -m venv .venv
)
call .venv\Scripts\activate.bat

python -m pip install -q polars pyarrow requests tqdm matplotlib psutil
if errorlevel 1 goto :error

python src\01_download.py
if errorlevel 1 goto :error
python src\02_storage.py
if errorlevel 1 goto :error
python src\03_quality.py
if errorlevel 1 goto :error
python src\04_build_panel.py
if errorlevel 1 goto :error

echo.
echo === XONG. Ket qua nam trong thu muc reports\ ===
pause
exit /b 0

:error
echo.
echo === CO LOI. Chup man hinh nay gui cho tro ly. ===
pause
exit /b 1

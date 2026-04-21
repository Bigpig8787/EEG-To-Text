@echo off
REM Train + Eval in one shot. Training ~hours, then auto-runs eval 4x.
REM If training fails (non-zero exit), eval is skipped.

echo ============================================================
echo [STAGE 1] Training
echo ============================================================
call "%~dp0train_multiview.bat"
if errorlevel 1 (
    echo [ERROR] Training failed with errorlevel %errorlevel%. Skipping eval.
    exit /b %errorlevel%
)

echo.
echo ============================================================
echo [STAGE 2] Evaluation
echo ============================================================
call "%~dp0eval_multiview.bat"

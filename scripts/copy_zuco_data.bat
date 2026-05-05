@echo off
REM Copy ZuCo .mat files from external source to project dataset folder.
REM Source layout (D:\EEG-BCI\Dataset_ZuCo):
REM   v1\task1- SR\Matlab files\          (note space in folder name)
REM   v1\task2 - NR\Matlab files\
REM   v1\task3 - TSR\Matlab files\
REM   v2\task2 - NR-2.0\Matlab files\
REM   v2\task2 - TSR\Matlab files\
REM Target layout (project dataset\ZuCo):
REM   task1-SR\Matlab files\
REM   task2-NR\Matlab files\
REM   task3-TSR\Matlab files\
REM   task2-NR-2.0\Matlab files\
REM   task2-TSR-2.0\Matlab files\

set SRC=D:\EEG-BCI\Dataset_ZuCo
set DST=%~dp0..\dataset\ZuCo

echo Source: %SRC%
echo Target: %DST%
echo.

REM /E    : copy subdirs incl. empty
REM /XO   : skip older (idempotent re-runs)
REM /R:1  : 1 retry on locked files
REM /W:1  : 1 sec wait
REM /NP   : no progress percent (cleaner log)
set RC_OPTS=/E /XO /R:1 /W:1 /NP

echo === task1-SR ===
robocopy "%SRC%\v1\task1- SR\Matlab files"      "%DST%\task1-SR\Matlab files"      %RC_OPTS%

echo === task2-NR ===
robocopy "%SRC%\v1\task2 - NR\Matlab files"     "%DST%\task2-NR\Matlab files"      %RC_OPTS%

echo === task3-TSR ===
robocopy "%SRC%\v1\task3 - TSR\Matlab files"    "%DST%\task3-TSR\Matlab files"     %RC_OPTS%

echo === task2-NR-2.0 ===
robocopy "%SRC%\v2\task2 - NR-2.0\Matlab files" "%DST%\task2-NR-2.0\Matlab files"  %RC_OPTS%

echo === task2-TSR-2.0 ===
robocopy "%SRC%\v2\task2 - TSR\Matlab files"    "%DST%\task2-TSR-2.0\Matlab files" %RC_OPTS%

echo.
echo Copy done. Next: run scripts\prepare_dataset.bat to build pickles.
pause

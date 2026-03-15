@echo off
REM Train all 5 folds with custom configuration
REM This script runs sequentially - each fold completes before starting the next

REM ===== CONFIGURE CONDA ENVIRONMENT =====
set CONDA_ENV=step-counter
REM =========================================

REM ===== CONFIGURE YOUR TRAINING HERE =====
set MODEL=shallow_cnn
set EPOCHS=50
set LR=0.001
set FILTERS=32
set DROPOUT=0.3
set PATIENCE=15
set BATCH_SIZE=32
REM =========================================

REM Activate conda environment
echo Activating conda environment: %CONDA_ENV%
call conda activate %CONDA_ENV%
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment '%CONDA_ENV%'
    echo Please create it first: conda create -n %CONDA_ENV% python=3.10
    pause
    exit /b 1
)
echo Environment activated successfully!
echo.

echo ========================================
echo Training All Folds - Step Counter CNN
echo ========================================
echo.
echo Configuration:
echo   Model: %MODEL%
echo   Epochs: %EPOCHS%
echo   Learning Rate: %LR%
echo   Filters: %FILTERS%
echo   Dropout: %DROPOUT%
echo   Patience: %PATIENCE%
echo   Batch Size: %BATCH_SIZE%
echo.
echo Starting training at %TIME%
echo This will take 1-2 hours...
echo.

REM Train each fold
for /L %%i in (0,1,4) do (
    echo.
    echo ========================================
    echo Training Fold %%i of 4
    echo ========================================
    python src\train.py --model %MODEL% --fold %%i --epochs %EPOCHS% --lr %LR% --n_filters %FILTERS% --dropout %DROPOUT% --patience %PATIENCE% --batch_size %BATCH_SIZE% --notes "Cross-validation training"

    if errorlevel 1 (
        echo ERROR: Fold %%i failed!
        pause
        exit /b 1
    )

    echo Fold %%i completed successfully!
)

echo.
echo ========================================
echo All folds trained successfully!
echo Finished at %TIME%
echo ========================================
echo.
echo Next steps:
echo   1. Evaluate results: python src\evaluate_cv.py --model shallow_cnn
echo   2. Check results in: models\saved\
echo.
pause

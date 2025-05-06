@echo off
chcp 65001 > nul
setlocal

REM Проверка, существует ли папка venv
if not exist "venv\Scripts\activate.bat" (
    cls
    echo Ошибка: Скрипт активации venv не найден по пути: venv\Scripts\activate.bat
    echo Убедитесь, что вы запускаете этот скрипт из корневой папки проекта FaceFusion,
    echo и что виртуальное окружение venv создано корректно.
    pause
    goto :eof
)

REM --- Определение текущего бэкенда ONNX Runtime ---
echo ==================================================
echo  Определение текущего бэкенда ONNX Runtime
echo ==================================================
echo.
echo import onnxruntime; print(onnxruntime.get_available_providers()) > detect_onnx_backend.py
venv\Scripts\python.exe detect_onnx_backend.py > onnx_backend.txt 2>&1

set "backend=unknown"
findstr /I "CUDAExecutionProvider" onnx_backend.txt > nul && set "backend=onnxruntime-gpu"
findstr /I "DmlExecutionProvider" onnx_backend.txt > nul && set "backend=onnxruntime-directml"
findstr /I "DirectMLExecutionProvider" onnx_backend.txt > nul && set "backend=onnxruntime-directml"

if "%backend%"=="onnxruntime-gpu" (
    echo Текущий бэкенд: onnxruntime-gpu (CUDA)
    del detect_onnx_backend.py
    del onnx_backend.txt
    echo.
    pause
    goto menu
)
if "%backend%"=="onnxruntime-directml" (
    echo Текущий бэкенд: onnxruntime-directml (DirectML)
    del detect_onnx_backend.py
    del onnx_backend.txt
    echo.
    pause
    goto menu
)
echo Не удалось определить бэкенд ONNX Runtime или он не установлен.
del detect_onnx_backend.py
del onnx_backend.txt
echo.
pause

:menu
cls
echo ==================================================
echo  Смена бэкенда ONNX Runtime
echo ==================================================
echo.
echo Текущий бэкенд определить сложно, выбирайте действие:
echo.
echo 1. Установить onnxruntime-gpu (для видеокарт NVIDIA CUDA)
echo 2. Установить onnxruntime-directml (для видеокарт AMD/Intel и старых NVIDIA)
echo 3. Выход
echo.
set /p choice="Введите номер опции (1, 2, 3): "

if "%choice%"=="1" goto install_gpu
if "%choice%"=="2" goto install_dml
if "%choice%"=="3" goto end

cls
echo Неверный выбор. Нажмите любую клавишу для возврата в меню...
pause > nul
goto menu

:install_gpu
cls
echo Деактивация текущего окружения (на всякий случай)...
call venv\Scripts\deactivate.bat > nul 2>&1
echo.
echo Активация виртуального окружения venv...
call venv\Scripts\activate.bat
echo.
echo Удаление onnxruntime-directml (если установлен)...
pip uninstall onnxruntime-directml -y
echo.
echo Установка onnxruntime-gpu...
pip install onnxruntime-gpu
if errorlevel 1 (
    echo.
    echo ОШИБКА: Не удалось установить onnxruntime-gpu. Проверьте интернет-соединение и логи pip.
    goto pause_exit
)
echo.
echo Успешно установлен бэкенд onnxruntime-gpu.
goto pause_exit

:install_dml
cls
echo Деактивация текущего окружения (на всякий случай)...
call venv\Scripts\deactivate.bat > nul 2>&1
echo.
echo Активация виртуального окружения venv...
call venv\Scripts\activate.bat
echo.
echo Удаление onnxruntime-gpu (если установлен)...
pip uninstall onnxruntime-gpu -y
echo.
echo Установка onnxruntime-directml...
pip install onnxruntime-directml
if errorlevel 1 (
    echo.
    echo ОШИБКА: Не удалось установить onnxruntime-directml. Проверьте интернет-соединение и логи pip.
    goto pause_exit
)
echo.
echo Успешно установлен бэкенд onnxruntime-directml.
goto pause_exit

:pause_exit
echo.
echo Нажмите любую клавишу для выхода...
pause > nul
goto end

:end
endlocal
exit /b 
@echo off
set pypath=home = %~dp0python
set venvpath=_ENV=%~dp0venv
if exist venv (powershell -command "$text = (gc venv\pyvenv.cfg) -replace 'home = .*', $env:pypath; $Utf8NoBomEncoding = New-Object System.Text.UTF8Encoding($False);[System.IO.File]::WriteAllLines('venv\pyvenv.cfg', $text, $Utf8NoBomEncoding);$text = (gc venv\scripts\activate.bat) -replace '_ENV=.*', $env:venvpath; $Utf8NoBomEncoding = New-Object System.Text.UTF8Encoding($False);[System.IO.File]::WriteAllLines('venv\scripts\activate.bat', $text, $Utf8NoBomEncoding);")

for /d %%i in (tmp\tmp*,tmp\pip*) do rd /s /q "%%i" 2>nul || ("%%i" && exit /b 1) & del /q tmp\tmp* > nul 2>&1 & rd /s /q pip\cache 2>nul

set appdata=%~dp0tmp
set userprofile=%~dp0tmp
set temp=%~dp0tmp

set BUILD_TOOLS_ROOT=%~dp0
set WindowsSDKDir=%BUILD_TOOLS_ROOT%Windows Kits\10
set VCToolsInstallDir=%BUILD_TOOLS_ROOT%VC\Tools\MSVC\14.42.34433
set WindowsSDKVersion=10.0.26100.0
set VSCMD_ARG_TGT_ARCH=x64
set VSCMD_ARG_HOST_ARCH=x64
set INCLUDE=%VCToolsInstallDir%\include;%WindowsSDKDir%\Include\%WindowsSDKVersion%\ucrt;%WindowsSDKDir%\Include\%WindowsSDKVersion%\shared;%WindowsSDKDir%\Include\%WindowsSDKVersion%\um;%WindowsSDKDir%\Include\%WindowsSDKVersion%\winrt;%WindowsSDKDir%\Include\%WindowsSDKVersion%\cppwinrt;
set LIB=%VCToolsInstallDir%\lib\%VSCMD_ARG_TGT_ARCH%;%WindowsSDKDir%\Lib\%WindowsSDKVersion%\ucrt\%VSCMD_ARG_TGT_ARCH%;%WindowsSDKDir%\Lib\%WindowsSDKVersion%\um\%VSCMD_ARG_TGT_ARCH%
set BUILD_TOOLS_BIN=%VCToolsInstallDir%\bin\Host%VSCMD_ARG_HOST_ARCH%\%VSCMD_ARG_TGT_ARCH%;%WindowsSDKDir%\bin\%WindowsSDKVersion%\%VSCMD_ARG_TGT_ARCH%;%WindowsSDKDir%\bin\%WindowsSDKVersion%\%VSCMD_ARG_TGT_ARCH%\ucrt

set PATH=%BUILD_TOOLS_BIN%;git\cmd;%~dp0python;%~dp0venv\scripts;%~dp0ffmpeg;%~dp0cmake;%~dp0cuda128\bin;%~dp0cuda128\lib;%~dp0curl\lib;%~dp0curl\bin;%~dp0tensorrt\lib

set CUDA_MODULE_LOADING=LAZY
set CUDA_PATH=%~dp0cuda128
set CUDNN_PATH=%~dp0cuda128

call venv\Scripts\activate.bat

python facefusion.py run --language EN
pause

REM Упаковано и собрано телеграм каналом Neutogen News: https://t.me/neurogen_news

@echo off
title Instalador Completo de Dependências
color 0A

echo ========================================
echo    INSTALADOR COMPLETO - Python & Pip
echo ========================================
echo.

echo Verificando se o Python está instalado...
python --version >nul 2>&1
if errorlevel 1 (
    echo Python não encontrado no PATH.
    echo Verificando com py launcher...
    py --version >nul 2>&1
    if errorlevel 1 (
        echo ERRO: Python não está instalado!
        echo Baixe em: https://www.python.org/downloads/
        echo Marque a opção "Add Python to PATH" durante a instalação.
        pause
        exit /b 1
    )
    set PYTHON_CMD=py
) else (
    set PYTHON_CMD=python
)

echo.
echo Python encontrado! Verificando pip...
%PYTHON_CMD% -m pip --version >nul 2>&1
if errorlevel 1 (
    echo Pip não encontrado. Instalando pip...
    echo.
    echo Baixando e instalando pip...
    %PYTHON_CMD% -m ensurepip --default-pip
    if errorlevel 1 (
        echo Tentando método alternativo...
        %PYTHON_CMD% -m pip install --upgrade pip
    )
)

echo.
echo Verificando versão do pip instalado...
%PYTHON_CMD% -m pip --version

echo.
echo Atualizando pip para a versão mais recente...
%PYTHON_CMD% -m pip install --upgrade pip

echo.
echo ========================================
echo INSTALANDO PACOTES DO PROJETO...
echo ========================================
echo.

echo [1/2] Instalando ultralytics, matplotlib e opencv-python...
%PYTHON_CMD% -m pip install ultralytics matplotlib opencv-python
if errorlevel 1 (
    echo ERRO na instalação do primeiro grupo de pacotes!
    pause
    exit /b 1
)

echo.
echo [2/2] Instalando torchreid...
%PYTHON_CMD% -m pip install torchreid
if errorlevel 1 (
    echo ERRO na instalação do torchreid!
    pause
    exit /b 1
)
echo.

echo Instalando módulos adicionais
echo gdown
%PYTHON_CMD% -m pip install gdown
echo tensorboard
%PYTHON_CMD% -m pip install tensorboard==2.11.1
echo scikit-learn
%PYTHON_CMD% -m pip install scikit-learn
echo tensorflow
%PYTHON_CMD% -m pip install tensorflow
%PYTHON_CMD% -m pip install numpy==1.24.2
%PYTHON_CMD% -m pip install cython==3.0.0
%PYTHON_CMD% -m pip install deep-sort-realtime
echo.
echo ========================================
echo INSTALAÇÃO CONCLUÍDA COM SUCESSO!
echo ========================================
echo.
echo Pacotes instalados:
%PYTHON_CMD% -m pip list | findstr "ultralytics matplotlib opencv torchreid"

echo Instalando MobileClip
cd ml-mobileclip-main
pip install .


echo.
pause
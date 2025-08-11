#!/bin/bash

# Script para ativar o ambiente virtual do SpA-Former
echo "Ativando ambiente virtual .venv_spa..."

# Verificar se estamos no diretório raiz do projeto
if [ ! -f "config.yml" ]; then
    echo "Erro: Execute este script do diretório raiz do projeto SpA-Former!"
    echo "Diretório atual: $(pwd)"
    exit 1
fi

# Verificar se o ambiente virtual existe
if [ ! -d ".venv_spa" ]; then
    echo "Erro: Ambiente virtual .venv_spa não encontrado!"
    echo "Execute: python3 -m venv .venv_spa"
    exit 1
fi

# Ativar o ambiente virtual
source .venv_spa/bin/activate

# Verificar se o PyTorch está instalado
python -c "import torch; print(f'✓ PyTorch {torch.__version__} instalado')" 2>/dev/null || {
    echo "❌ PyTorch não encontrado. Instale as dependências:"
    echo "pip install -r .dev/requirements.txt"
    exit 1
}

# Verificar CUDA
python -c "import torch; print(f'✓ CUDA disponível: {torch.cuda.is_available()}')" 2>/dev/null || {
    echo "❌ Erro ao verificar CUDA"
    exit 1
}

echo "✓ Ambiente virtual ativado com sucesso!"
echo "Para desativar: deactivate"
echo ""
echo "Comandos úteis:"
echo "  - Testar modelo: python demo.py --test_filepath <imagem> --pretrained <modelo> --cuda"
echo "  - Treinar modelo: python train.py"
echo "  - Predição: python predict.py --config config.yml --test_dir <pasta> --out_dir <saida> --pretrained <modelo> --cuda"
echo ""
echo "📚 Documentação: .dev/doc/ENVIRONMENT_SETUP.md" 
#!/bin/bash

# Script para baixar modelos pré-treinados do SpA-Former

echo "📥 Baixando modelos pré-treinados do SpA-Former..."

# Criar pasta se não existir
mkdir -p pretrained_models

# URLs dos modelos (você pode adicionar mais URLs aqui)
MODEL_URLS=(
    "https://drive.google.com/uc?id=1pxwwAfwnGKkLj-GAlkVCevbEQM4basgR"
    "https://huggingface.co/datasets/SpA-Former/models/resolve/main/gen_model_epoch_200.pth"
)

echo "🔗 Links para download manual:"
echo "1. Google Drive: https://drive.google.com/drive/folders/1pxwwAfwnGKkLj-GAlkVCevbEQM4basgR?usp=sharing"
echo "2. Baidu Drive: https://pan.baidu.com/s/1slny1G_9WuxBcoyw5eKUVA (提取码：rpis)"
echo ""
echo "📋 Instruções:"
echo "1. Baixe o arquivo 'gen_model_epoch_200.pth' de um dos links acima"
echo "2. Coloque o arquivo na pasta 'pretrained_models/'"
echo "3. Execute: python demo.py --test_filepath <imagem> --pretrained pretrained_models/gen_model_epoch_200.pth --cuda"
echo ""

# Tentar baixar usando curl se disponível
if command -v curl &> /dev/null; then
    echo "🔄 Tentando baixar modelo via curl..."
    curl -L -o pretrained_models/gen_model_epoch_200.pth "https://huggingface.co/datasets/SpA-Former/models/resolve/main/gen_model_epoch_200.pth" 2>/dev/null && {
        echo "✅ Modelo baixado com sucesso!"
        echo "📍 Localização: pretrained_models/gen_model_epoch_200.pth"
    } || {
        echo "❌ Falha no download automático."
        echo "📥 Faça o download manual dos links acima."
    }
else
    echo "❌ curl não encontrado. Faça o download manual dos links acima."
fi

echo ""
echo "🎯 Para testar após baixar o modelo:"
echo "python demo.py --test_filepath /home/consultoria/Desktop/shadow_test1/20250715_01_00017_PAN.jpeg --pretrained pretrained_models/gen_model_epoch_200.pth --cuda" 
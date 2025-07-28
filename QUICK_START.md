# 🚀 SpA-Former - Início Rápido

## 📋 Pré-requisitos

- Python 3.10+
- GPU NVIDIA com CUDA (recomendado)
- Git

## ⚡ Configuração Rápida

### 1. Ativar ambiente virtual
```bash
.dev/activate_env.sh
```

### 2. Baixar modelo pré-treinado
- [Google Drive](https://drive.google.com/drive/folders/1pxwwAfwnGKkLj-GAlkVCevbEQM4basgR?usp=sharing)
- [Baidu Drive](https://pan.baidu.com/s/1slny1G_9WuxBcoyw5eKUVA)

### 3. Testar modelo
```bash
python demo.py --test_filepath <imagem> --pretrained <modelo> --cuda
```

## 📚 Documentação Completa

Para informações detalhadas sobre configuração, treinamento e uso:
- **Configuração do ambiente**: `.dev/doc/ENVIRONMENT_SETUP.md`
- **README original**: `README.md`

## 🎯 Comandos Principais

| Comando | Descrição |
|---------|-----------|
| `.dev/activate_env.sh` | Ativar ambiente virtual |
| `python demo.py --test_filepath <img> --pretrained <model> --cuda` | Teste rápido |
| `python train.py` | Treinar modelo |
| `python predict.py --config config.yml --test_dir <dir> --out_dir <out> --pretrained <model> --cuda` | Predição em lote |

## 🔧 Estrutura do Projeto

```
SpA-Former-shadow-removal/
├── .dev/                    # 🛠️ Arquivos de desenvolvimento
│   ├── activate_env.sh      # Script de ativação
│   ├── requirements.txt     # Dependências
│   └── doc/                 # 📚 Documentação
├── .venv_spa/              # 🐍 Ambiente virtual
├── config.yml              # ⚙️ Configurações
├── train.py                # 🎓 Treinamento
├── predict.py              # 🔮 Predição
├── demo.py                 # 🎮 Demonstração
└── SpA_Former.py           # 🧠 Arquitetura do modelo
```

## 🆘 Problemas Comuns

### Ambiente não ativa
```bash
# Recriar ambiente
rm -rf .venv_spa
python3 -m venv .venv_spa
source .venv_spa/bin/activate
pip install -r .dev/requirements.txt
```

### CUDA não funciona
```bash
# Reinstalar PyTorch com CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 📞 Suporte

- **Código**: framebreak@sjtu.edu.cn
- **Documentação**: `.dev/doc/ENVIRONMENT_SETUP.md`

---

**SpA-Former**: Transformer eficiente e leve para remoção de sombras em imagens (IJCNN 2023) 
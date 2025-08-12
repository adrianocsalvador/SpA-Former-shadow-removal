# Configuração do Ambiente Virtual SpA-Former

## ✅ Ambiente Virtual Criado

O ambiente virtual `.venv_spa` foi criado com sucesso com as seguintes configurações:

- **Python**: 3.10.12
- **PyTorch**: 2.5.1+cu121 (com suporte a CUDA)
- **GPU**: NVIDIA GeForce RTX 4060 (CUDA 12.9)

## 🚀 Como usar

### Ativação rápida
```bash
# Execute do diretório raiz do projeto
.dev/activate_env.sh
```

### Ativação manual
```bash
source .venv_spa/bin/activate
```

### Desativação
```bash
deactivate
```

## 📦 Dependências instaladas

### Principais bibliotecas:
- `torch==2.5.1+cu121` - PyTorch com suporte a CUDA
- `torchvision==0.20.1+cu121` - Visão computacional
- `torchaudio==2.5.1+cu121` - Processamento de áudio
- `opencv-python==4.12.0.88` - Processamento de imagens
- `matplotlib==3.10.3` - Visualização
- `numpy==2.1.2` - Computação numérica
- `tqdm==4.67.1` - Barras de progresso
- `pyyaml==6.0.2` - Configurações YAML
- `attrdict==2.0.1` - Dicionários com atributos

### Bibliotecas CUDA:
- `nvidia-cudnn-cu12==9.1.0.70`
- `nvidia-cublas-cu12==12.1.3.1`
- `nvidia-cufft-cu12==11.0.2.54`
- E outras bibliotecas NVIDIA para aceleração GPU

## 🔧 Verificação do ambiente

Para verificar se tudo está funcionando:

```bash
# Ativar o ambiente
source .venv_spa/bin/activate

# Verificar PyTorch e CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## 📁 Estrutura do projeto

```
SpA-Former-shadow-removal/
├── .venv_spa/              # Ambiente virtual (ignorado pelo git)
├── .dev/                    # Pasta de desenvolvimento (ignorada pelo git)
│   ├── activate_env.sh      # Script de ativação
│   ├── requirements.txt     # Lista de dependências
│   └── doc/                 # Documentação
│       └── ENVIRONMENT_SETUP.md
├── config.yml              # Configurações do modelo
├── train.py                # Script de treinamento
├── predict.py              # Script de predição
├── demo.py                 # Demonstração rápida
├── SpA_Former.py           # Arquitetura do modelo
├── .gitignore              # Arquivos ignorados pelo git
└── data/                   # Dataset (a ser configurado)
```

## 🎯 Próximos passos

1. **Baixar modelo pré-treinado**:
   - [Google Drive](https://drive.google.com/drive/folders/1pxwwAfwnGKkLj-GAlkVCevbEQM4basgR?usp=sharing)
   - [Baidu Drive](https://pan.baidu.com/s/1slny1G_9WuxBcoyw5eKUVA)

2. **Testar o modelo**:
   ```bash
   python demo.py --test_filepath <imagem> --pretrained <modelo> --cuda
   ```

3. **Configurar dataset** (se necessário):
   - Baixar ISTD dataset
   - Organizar em `data/ISTD/train/` e `data/ISTD/test/`

## 🆘 Solução de problemas

### Se PyTorch não encontrar CUDA:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Se houver problemas de memória GPU:
- Reduza o batch size no `config.yml`
- Use `--no-cuda` para CPU (mais lento)

### Para reinstalar dependências:
```bash
pip install -r .dev/requirements.txt
```

### Para criar novo ambiente virtual:
```bash
# Remover ambiente atual
rm -rf .venv_spa

# Criar novo ambiente
python3 -m venv .venv_spa

# Ativar e instalar dependências
source .venv_spa/bin/activate
pip install -r .dev/requirements.txt
```

## 📞 Suporte

Para dúvidas sobre o código: **framebreak@sjtu.edu.cn**

## 🔒 Controle de versão

Os seguintes arquivos/pastas são ignorados pelo git:
- `.venv_spa/` - Ambiente virtual
- `.dev/` - Arquivos de desenvolvimento
- `*.pth` - Modelos treinados
- `results/`, `out/`, `output/` - Resultados
- `data/`, `datasets/` - Datasets 
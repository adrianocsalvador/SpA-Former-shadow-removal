# 🪟 Sistema de Janela Deslizante para SpA-Former

Este sistema implementa treinamento com **janela deslizante em tempo real** para o modelo SpA-Former de remoção de sombras. Em vez de usar imagens pré-processadas, o sistema carrega imagens originais completas e cria janelas deslizantes dinamicamente durante o treinamento.

## 🎯 Características Principais

- **Processamento em tempo real**: Janelas são criadas dinamicamente durante o treinamento
- **Flexibilidade**: Configurável tamanho de janela e stride
- **Eficiência de memória**: Processa apenas as janelas necessárias
- **Data augmentation**: Suporte para rotação, flip e crop aleatório
- **Visualização**: Scripts para visualizar o processo de janela deslizante

## 📁 Arquivos do Sistema

```
sliding_window_train.py          # Script principal de treinamento
start_sliding_window_training.py # Script de inicialização
demo_sliding_window.py          # Script de demonstração
config_sliding_window.yml       # Configuração específica
README_SLIDING_WINDOW.md        # Esta documentação
```

## 🚀 Como Usar

### 1. Configuração Inicial

Certifique-se de que o dataset ISTD está configurado corretamente:

```
data/ISTD/train/
├── train_A/          # Imagens com sombra
├── train_C/          # Imagens sem sombra (ground truth)
└── train_list.txt    # Lista de imagens de treinamento
```

### 2. Executar Demonstração

Para visualizar como funciona o sistema de janela deslizante:

```bash
# Demonstração com imagem específica
python demo_sliding_window.py --image 100-6.png

# Testar o dataset
python demo_sliding_window.py --test_dataset

# Usar configurações personalizadas
python demo_sliding_window.py --window_size 512 384 --stride 256 192
```

### 3. Iniciar Treinamento

```bash
# Treinamento básico
python start_sliding_window_training.py

# Com parâmetros personalizados
python start_sliding_window_training.py \
    --window_size 640 480 \
    --stride 320 240 \
    --batch_size 4 \
    --epochs 100 \
    --lr 0.0004

# Com modelo pré-treinado
python start_sliding_window_training.py \
    --pretrained_gen results_original/models/gen_model_epoch_50.pth
```

### 4. Treinamento Direto

```bash
# Usar configuração personalizada
python sliding_window_train.py
```

## ⚙️ Configurações

### Parâmetros da Janela Deslizante

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `window_width` | Largura da janela | 640 |
| `window_height` | Altura da janela | 480 |
| `stride_width` | Stride horizontal | 320 |
| `stride_height` | Stride vertical | 240 |

### Parâmetros de Treinamento

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `batchsize` | Tamanho do batch | 4 |
| `epoch` | Número de épocas | 100 |
| `lr` | Learning rate | 0.0004 |
| `lamb` | Peso da loss L1 | 100 |

## 🔧 Como Funciona

### 1. Carregamento de Imagens
- O sistema carrega imagens originais completas do dataset
- Não há pré-processamento ou redimensionamento

### 2. Geração de Janelas
- Para cada imagem, são geradas janelas deslizantes
- O stride determina a sobreposição entre janelas
- Janelas finais são criadas para cobrir bordas da imagem

### 3. Processamento em Tempo Real
- Janelas são criadas dinamicamente durante o treinamento
- Cada batch contém janelas de diferentes posições
- O sistema move automaticamente para a próxima imagem quando todas as janelas são processadas

### 4. Cálculo da Máscara de Sombra
- A máscara é calculada como: `M = clip((target - input).sum(axis=2), 0, 1)`
- Isso identifica automaticamente as áreas com sombra

## 📊 Visualização

O script `demo_sliding_window.py` cria visualizações que mostram:

1. **Imagem Original**: Com sombra
2. **Imagem Target**: Sem sombra (ground truth)
3. **Máscara de Sombra**: Diferença entre as imagens
4. **Janelas Extraídas**: Exemplos de janelas deslizantes

## 🎛️ Configurações Avançadas

### Data Augmentation

```yaml
enable_data_augmentation: True
random_crop: True
horizontal_flip: True
vertical_flip: False
rotation_range: 10  # graus
```

### Otimizações de Memória

```yaml
# Para GPUs com menos memória
batchsize: 2
window_width: 512
window_height: 384
stride_width: 256
stride_height: 192
```

### Para GPUs Potentes (24GB+)

```yaml
# Para GPUs com muita memória
batchsize: 8
window_width: 1024
window_height: 768
stride_width: 512
stride_height: 384
```

## 🔍 Monitoramento

O sistema gera logs detalhados incluindo:

- **Loss por iteração**: Generator e discriminator losses
- **Tempo por época**: Tempo de processamento
- **Validação**: Métricas de validação por época
- **Checkpoints**: Modelos salvos periodicamente

## 🐛 Solução de Problemas

### Erro: "Dataset não encontrado"
```bash
# Verificar estrutura do dataset
ls -la data/ISTD/train/
# Deve conter: train_A/, train_C/, train_list.txt
```

### Erro: "Out of memory"
```bash
# Reduzir batch size e tamanho da janela
python start_sliding_window_training.py --batch_size 2 --window_size 512 384
```

### Erro: "Imagens não encontradas"
```bash
# Verificar se as imagens existem
ls data/ISTD/train/train_A/ | head -5
ls data/ISTD/train/train_C/ | head -5
```

## 📈 Vantagens do Sistema

1. **Flexibilidade**: Funciona com imagens de qualquer tamanho
2. **Eficiência**: Não precisa pré-processar todas as imagens
3. **Memória**: Usa apenas a memória necessária para as janelas atuais
4. **Qualidade**: Preserva a qualidade original das imagens
5. **Escalabilidade**: Fácil de adaptar para diferentes tamanhos de GPU

## 🔬 Experimentos Sugeridos

1. **Comparar diferentes strides**:
   ```bash
   # Stride pequeno (mais sobreposição)
   python start_sliding_window_training.py --stride 160 120
   
   # Stride grande (menos sobreposição)
   python start_sliding_window_training.py --stride 480 360
   ```

2. **Testar diferentes tamanhos de janela**:
   ```bash
   # Janelas pequenas
   python start_sliding_window_training.py --window_size 512 384
   
   # Janelas grandes
   python start_sliding_window_training.py --window_size 1024 768
   ```

3. **Ajustar learning rate**:
   ```bash
   # Learning rate menor
   python start_sliding_window_training.py --lr 0.0001
   
   # Learning rate maior
   python start_sliding_window_training.py --lr 0.001
   ```

## 📞 Suporte

Para dúvidas ou problemas:
- Verifique os logs de erro
- Teste com configurações menores primeiro
- Use o script de demonstração para verificar o dataset

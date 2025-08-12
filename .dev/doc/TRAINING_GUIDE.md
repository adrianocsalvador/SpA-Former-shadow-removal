# 🎓 Guia de Treinamento SpA-Former

## 📋 Pré-requisitos

- Dataset com pares de imagens (com sombra + sem sombra)
- GPU NVIDIA com pelo menos 8GB de VRAM (recomendado)
- Tempo disponível (treinamento pode levar 2-24 horas)

## 🗂️ Estrutura do Dataset

Seu dataset deve ter esta estrutura:
```
seu_dataset/
├── imagens_com_sombra/     # Pasta com imagens COM sombra
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── imagens_sem_sombra/     # Pasta com imagens SEM sombra (ground truth)
    ├── img001.jpg
    ├── img002.jpg
    └── ...
```

**Importante**: As imagens devem ter nomes correspondentes!

## 🚀 Passo a Passo

### 1. Configurar Dataset

```bash
# Ativar ambiente virtual
.dev/activate_env.sh

# Configurar dataset
python .dev/setup_training.py \
    --source_dir "/caminho/para/imagens_com_sombra" \
    --target_dir "/caminho/para/imagens_sem_sombra" \
    --output_dir "./data_custom" \
    --epochs 100 \
    --batch_size 4
```

### 2. Verificar Configuração

O script criará:
- `./data_custom/ISTD/train/input/` - Imagens com sombra para treino
- `./data_custom/ISTD/train/target/` - Imagens sem sombra para treino
- `./data_custom/ISTD/test/input/` - Imagens com sombra para teste
- `./data_custom/ISTD/test/target/` - Imagens sem sombra para teste
- `./data_custom/config_custom.yml` - Configuração personalizada

### 3. Iniciar Treinamento

```bash
python .dev/start_training.py --config ./data_custom/config_custom.yml
```

## ⚙️ Configurações Importantes

### Tamanho das Imagens
- **Recomendado**: 640x480 pixels
- **Mínimo**: 256x256 pixels
- **Máximo**: 1024x1024 pixels (depende da GPU)

### Parâmetros de Treinamento
- **Épocas**: 50-200 (mais dados = menos épocas)
- **Batch Size**: 2-8 (depende da memória da GPU)
- **Learning Rate**: 0.0004 (padrão)

### Ajustes para GPU com Menos Memória
```yaml
batchsize: 2          # Reduzir batch size
width: 480            # Reduzir largura
height: 640           # Reduzir altura
```

## 📊 Monitoramento

Durante o treinamento, você verá:
- **Loss**: Deve diminuir ao longo do tempo
- **PSNR/SSIM**: Métricas de qualidade (maior = melhor)
- **Tempo por época**: Para estimar tempo total

### Logs Salvos
- `./results_custom/log.txt` - Log detalhado
- `./results_custom/models/` - Modelos salvos a cada época

## 🎯 Resultados Esperados

### Boas Práticas
- **Mínimo 100 pares** de imagens para treinamento
- **Diversidade**: Diferentes tipos de sombra, iluminação, cenários
- **Qualidade**: Imagens nítidas, bem alinhadas
- **Consistência**: Mesmo tamanho e formato

### Tempo de Treinamento Estimado
| Dataset | GPU | Tempo Estimado |
|---------|-----|----------------|
| 100 imagens | RTX 4060 | 2-4 horas |
| 500 imagens | RTX 4060 | 6-12 horas |
| 1000 imagens | RTX 4060 | 12-24 horas |

## 🔧 Solução de Problemas

### Erro de Memória GPU
```bash
# Reduzir batch size
python .dev/setup_training.py --batch_size 2

# Ou usar CPU (muito mais lento)
# Editar config_custom.yml: cuda: False
```

### Dataset Muito Pequeno
- Aumentar épocas (200-500)
- Usar data augmentation
- Considerar transfer learning

### Treinamento Muito Lento
- Reduzir tamanho das imagens
- Aumentar batch size (se memória permitir)
- Usar GPU mais potente

## 📈 Avaliação

### Durante Treinamento
- Monitore as métricas PSNR/SSIM
- Verifique imagens de validação
- Pare se loss não diminuir por 10 épocas

### Após Treinamento
```bash
# Testar modelo treinado
python .dev/test_demo.py \
    --test_filepath "sua_imagem.jpg" \
    --pretrained "./results_custom/models/gen_model_epoch_100.pth" \
    --cuda
```

## 💡 Dicas Avançadas

### Transfer Learning
```bash
# Usar modelo pré-treinado como inicialização
# Editar config_custom.yml:
gen_init: models/gen/gen_model_epoch_160.pth
```

### Fine-tuning
- Treinar com learning rate menor (0.0001)
- Congelar algumas camadas iniciais
- Usar dataset específico do domínio

### Data Augmentation
- Rotação, flip horizontal
- Ajuste de brilho/contraste
- Ruído gaussiano

## 🆘 Suporte

Se encontrar problemas:
1. Verifique logs em `./results_custom/log.txt`
2. Confirme estrutura do dataset
3. Teste com dataset menor primeiro
4. Verifique memória GPU disponível

---

**Lembre-se**: Treinamento de modelos de deep learning requer paciência e experimentação. Comece com datasets pequenos e vá aumentando gradualmente! 🚀 
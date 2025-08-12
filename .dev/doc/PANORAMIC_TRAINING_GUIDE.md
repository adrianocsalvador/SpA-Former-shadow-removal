# 🎓 Guia de Treinamento SpA-Former - Imagens Panorâmicas Ladybug6

## 📋 Especificidades do Dataset

### Estrutura das Imagens
- **Formato original**: 12288 x 6144 pixels (panorâmico 2:1)
- **Câmera**: Ladybug6
- **Organização**: Por missões com pastas PAN e PAN_SOMBRA

### Estrutura de Pastas
```
/mnt/48EC7EE9EC7EE9EC7ED0A4/Teste_Sombra/
├── 20250717_01/
│   ├── PAN/           # Imagens panorâmicas COM sombra
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── PAN_SOMBRA/    # Imagens panorâmicas SEM sombra (ground truth)
│       ├── img001.jpg
│       ├── img002.jpg
│       └── ...
└── 20250717_02/
    ├── PAN/
    └── PAN_SOMBRA/
```

## 🚀 Configuração Automática

### 1. Configurar Dataset Panorâmico

```bash
# Ativar ambiente virtual
.dev/activate_env.sh

# Configurar dataset panorâmico
python .dev/setup_panoramic_dataset.py \
    --base_dir "/mnt/48EC7EE9EC7ED0A4/Teste_Sombra/" \
    --output_dir "./data_panoramic" \
    --width 640 \
    --height 480 \
    --epochs 100 \
    --batch_size 4
```

### 2. O que o script faz automaticamente:

✅ **Detecta missões** (20250717_01, 20250717_02, etc.)  
✅ **Encontra pares** de imagens correspondentes  
✅ **Redimensiona** para 640x480 mantendo proporção  
✅ **Adiciona padding** se necessário  
✅ **Organiza** em treino/teste (80%/20%)  
✅ **Cria configuração** personalizada  

### 3. Iniciar Treinamento

```bash
python .dev/start_training.py --config ./data_panoramic/config_panoramic.yml
```

## 📐 Processamento de Imagens

### Redimensionamento Inteligente
- **Mantém proporção** original (2:1 panorâmico)
- **Centraliza** a imagem no frame 640x480
- **Adiciona padding** preto se necessário
- **Preserva** detalhes importantes

### Exemplo de Redimensionamento
```
Original: 12288 x 6144 (2:1)
Redimensionado: 640 x 320 (mantém 2:1)
Final: 640 x 480 (com padding 80px top/bottom)
```

## ⚙️ Configurações Otimizadas

### Para Imagens Panorâmicas
```yaml
width: 640          # Largura padrão
height: 480         # Altura padrão
batchsize: 4        # Ajustar conforme memória GPU
epoch: 100          # Mais épocas para datasets pequenos
```

### Ajustes para Memória Limitada
```yaml
batchsize: 2        # Reduzir batch size
width: 512          # Reduzir largura
height: 384         # Reduzir altura proporcionalmente
```

## 🎯 Teste com Imagens Panorâmicas

### Teste Individual
```bash
python .dev/test_panoramic.py \
    --test_filepath "/caminho/para/sua/imagem_panoramica.jpg" \
    --pretrained "./results_panoramic/models/gen_model_epoch_100.pth" \
    --cuda \
    --show
```

### Características do Teste
- **Redimensionamento automático** para 640x480
- **Preserva proporção** panorâmica
- **Mostra resultado** lado a lado (original | resultado | attention)
- **Salva automaticamente** com sufixo `_panoramic_result`

## 📊 Monitoramento do Treinamento

### Métricas Importantes
- **Loss**: Deve diminuir consistentemente
- **PSNR**: Qualidade da reconstrução (maior = melhor)
- **SSIM**: Similaridade estrutural (maior = melhor)

### Logs Salvos
- `./results_panoramic/log.txt` - Log detalhado
- `./results_panoramic/models/` - Modelos salvos
- `./results_panoramic/images/` - Imagens de validação

## 🔧 Solução de Problemas Específicos

### Erro de Memória com Imagens Grandes
```bash
# Reduzir tamanho de processamento
python .dev/setup_panoramic_dataset.py \
    --width 512 \
    --height 384 \
    --batch_size 2
```

### Dataset Muito Pequeno
- **Aumentar épocas**: 200-500
- **Usar transfer learning**: Inicializar com modelo pré-treinado
- **Data augmentation**: Rotação, flip horizontal

### Qualidade de Redimensionamento
- **Problema**: Imagens muito pequenas após redimensionamento
- **Solução**: Aumentar tamanho final (ex: 1024x768)

## 💡 Dicas para Imagens Panorâmicas

### Boas Práticas
1. **Alinhamento**: Certifique-se de que PAN e PAN_SOMBRA estão alinhados
2. **Qualidade**: Use imagens nítidas e bem expostas
3. **Diversidade**: Inclua diferentes tipos de sombra e iluminação
4. **Consistência**: Mantenha mesmo formato e resolução

### Otimizações
- **Crop inteligente**: Focar em áreas com sombras
- **Multi-scale**: Treinar com diferentes resoluções
- **Attention maps**: Analisar onde o modelo foca

## 📈 Resultados Esperados

### Com Dataset Panorâmico
- **Melhor generalização** para imagens panorâmicas
- **Preservação de proporções** originais
- **Remoção eficiente** de sombras em cenas amplas

### Tempo de Treinamento Estimado
| Missões | Imagens | GPU | Tempo |
|---------|---------|-----|-------|
| 2 | ~50 | RTX 4060 | 4-8 horas |
| 5 | ~100 | RTX 4060 | 8-16 horas |
| 10+ | ~200+ | RTX 4060 | 16-24 horas |

## 🆘 Suporte Específico

### Problemas Comuns
1. **Imagens não encontradas**: Verificar estrutura de pastas
2. **Memória insuficiente**: Reduzir batch size ou tamanho
3. **Qualidade ruim**: Verificar alinhamento das imagens
4. **Treinamento lento**: Usar GPU mais potente

### Comandos Úteis
```bash
# Verificar estrutura do dataset
ls -la /mnt/48EC7EE9EC7ED0A4/Teste_Sombra/

# Contar imagens em cada missão
find /mnt/48EC7EE9EC7ED0A4/Teste_Sombra/ -name "*.jpg" | wc -l

# Verificar tamanho das imagens
file /mnt/48EC7EE9EC7ED0A4/Teste_Sombra/*/PAN/*.jpg | head -5
```

---

**Lembre-se**: Imagens panorâmicas têm características únicas. O modelo treinado será especializado em remover sombras em cenas amplas e panorâmicas! 🌅 
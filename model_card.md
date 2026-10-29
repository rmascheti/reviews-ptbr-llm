---
base_model: Qwen/Qwen2.5-0.5B
library_name: peft
license: mit
language:
- pt
pipeline_tag: text-classification
tags:
- lora
- peft
- sentiment-analysis
- portuguese
- reviews
datasets:
- ruanchaves/b2w-reviews01
metrics:
- f1
- accuracy
model-index:
- name: qwen2.5-0.5b-reviews-ptbr
  results:
  - task:
      type: text-classification
      name: Sentiment classification
    dataset:
      name: B2W-Reviews01 (test split, 12.689 reviews)
      type: ruanchaves/b2w-reviews01
    metrics:
    - type: f1
      name: Macro-F1
      value: 0.7865
    - type: accuracy
      value: 0.8781
---

# Qwen2.5-0.5B + LoRA · sentimento de reviews em português

Adapter LoRA que transforma o `Qwen/Qwen2.5-0.5B` num classificador de sentimento (negativo / neutro / positivo) para reviews de e-commerce em português do Brasil.

Código, dados e análise completa: [github.com/rmascheti/reviews-ptbr-llm](https://github.com/rmascheti/reviews-ptbr-llm)

## Resultados

Test set de 12.689 reviews do B2W-Reviews01, comparado com um baseline TF-IDF + Regressão Logística avaliado na mesma execução:

| Modelo | Macro-F1 | Acurácia |
|---|---|---|
| TF-IDF + Regressão Logística | 0,757 | 83,7% |
| **Este modelo** | **0,787** | **87,8%** |

| Classe | Precision | Recall | F1 |
|---|---|---|---|
| negativo | 0,895 | 0,944 | 0,918 |
| neutro | 0,572 | 0,467 | 0,514 |
| positivo | 0,921 | 0,934 | 0,927 |

O modelo erra 25% menos que o baseline. A classe neutra continua difícil (F1 0,51): os rótulos vêm da nota, e reviews com nota 3 misturam elogios e reclamações.

## Como usar

```python
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

labels = ["negativo", "neutro", "positivo"]
tok = AutoTokenizer.from_pretrained("rmascheti/qwen2.5-0.5b-reviews-ptbr")
base = AutoModelForSequenceClassification.from_pretrained("Qwen/Qwen2.5-0.5B", num_labels=3)
base.config.pad_token_id = tok.pad_token_id
model = PeftModel.from_pretrained(base, "rmascheti/qwen2.5-0.5b-reviews-ptbr").eval()

texts = ["Produto excelente, chegou antes do prazo.", "Bom, mas a entrega atrasou uma semana."]
with torch.no_grad():
    probs = model(**tok(texts, padding=True, return_tensors="pt")).logits.softmax(-1)
print([labels[i] for i in probs.argmax(-1)])  # ['positivo', 'neutro']
```

O adapter inclui a cabeça de classificação treinada (`modules_to_save`), então não é preciso treinar nada para usar.

## Treino

- **Dados:** B2W-Reviews01 (Americanas S.A., CC-BY-4.0). Título + texto, sentimento derivado da nota (1–2 negativo, 3 neutro, 4–5 positivo), deduplicado, split estratificado 80/10/10 (seed 42). Treino em 40 mil exemplos estratificados.
- **LoRA:** r=16, α=32, dropout 0,05, em `q/k/v/o_proj` e `gate/up/down_proj`. 8,8M parâmetros treináveis (1,75%).
- **Otimização:** 1 época, lr 2e-4, cosine com 5% de warmup, batch efetivo 32, fp16, max_len 256.
- **Seleção:** checkpoint do passo 1000 de 1251, pelo maior macro-F1 na validação.
- **Hardware:** 2× Tesla T4 (Kaggle), 46 min de treino, pico de 8,3 GB por GPU.

## Limitações

- Rótulos derivados da nota, não anotados à mão.
- Treinado em reviews de produtos; não foi avaliado em outros domínios (restaurantes, serviços, redes sociais).
- Uma única execução, sem intervalo de confiança.
- Cerca de 30 ms por review em GPU (vazão em lote), contra 0,05 ms do baseline em CPU. Para volume alto, avalie se o ganho compensa o custo.

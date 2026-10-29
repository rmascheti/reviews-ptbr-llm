# Reviews PT-BR com LLM pequeno: quando o fine-tuning vale o custo?

Classificação de sentimento (negativo / neutro / positivo) em **~127 mil reviews reais de e-commerce em português** (B2W-Reviews01), comparando um baseline clássico com um LLM de 0,5B parâmetros ajustado com **LoRA**.

A pergunta de negócio: para uma empresa que processa milhões de avaliações, **quanto de qualidade um LLM fine-tunado entrega a mais, e a que custo de latência/infra**?

## Resultados (test set, 12.689 reviews)

| Modelo | Macro-F1 | Acurácia | F1 "neutro" | Latência | Treino |
|---|---|---|---|---|---|
| TF-IDF + Regressão Logística | **0.753** | 0.835 | 0.474 | 0.03 ms/review (CPU) | 32 s (CPU) |
| Qwen2.5-0.5B + LoRA r=16 | _a preencher_ | | | | |

O baseline já é forte nos extremos (F1 ≈ 0,89), mas erra muito no **neutro**: notas 3 misturam elogio e reclamação ("produto bom, mas a entrega atrasou"). É ali que um modelo que entende contexto deve fazer diferença, e é a métrica a acompanhar.

## Abordagem

1. **Dados** (`src/prepare_data.py`): título + texto do review, sentimento derivado da nota (1–2, 3, 4–5), deduplicação, split estratificado 80/10/10 com seed fixa.
2. **Baseline** (`src/baseline.py`): TF-IDF de uni+bigramas, regressão logística com `class_weight="balanced"`.
3. **LLM** (`src/train_lora.py`): `Qwen2.5-0.5B` com cabeça de classificação, LoRA em todas as projeções de atenção e MLP (8,8M parâmetros treináveis, 1,75% do total), fp16 numa T4 gratuita do Kaggle, melhor checkpoint por macro-F1 na validação.

## Como reproduzir

```bash
uv sync
curl -sL -o data/B2W-Reviews01.csv https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/main/B2W-Reviews01.csv
uv run python src/prepare_data.py
uv run python src/baseline.py
uv run --group train python src/train_lora.py --smoke   # valida o pipeline em CPU
```

Treino completo: `notebooks/kaggle_train.ipynb` (GPU T4 do Kaggle). O adapter treinado será publicado em [huggingface.co/rmascheti/qwen2.5-0.5b-reviews-ptbr](https://huggingface.co/rmascheti/qwen2.5-0.5b-reviews-ptbr).

## Próximos passos

- [ ] Destilação: usar um LLM maior como "professor" para rotular o **motivo** da reclamação (entrega, defeito, produto errado, atendimento) e treinar o modelo pequeno nisso
- [ ] Demo no Hugging Face Spaces (Gradio)
- [ ] Análise de erros: exemplos em que os dois modelos discordam

## Dados

[B2W-Reviews01](https://github.com/americanas-tech/b2w-reviews01) (Americanas S.A.), licença CC-BY-4.0. Real, Gabriel et al., *B2W-Reviews01: An open product reviews corpus*, STIL 2019.

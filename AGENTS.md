# AGENTS.md

Instruções para agentes de IA que trabalham neste projeto.

## Projeto

Classificação de sentimento (negativo / neutro / positivo) em reviews PT-BR do B2W-Reviews01, comparando TF-IDF + Regressão Logística com Qwen2.5-0.5B ajustado via LoRA. A pergunta central é custo-benefício: quanto o LLM melhora e quanto custa a mais. Ver `README.md` e `DECISIONS.md`.

## Comandos

```bash
uv sync                                                    # deps base (pandas, sklearn)
curl -sL -o data/B2W-Reviews01.csv https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/main/B2W-Reviews01.csv
uv run python src/prepare_data.py                          # gera data/{train,val,test}.parquet
uv run python src/baseline.py                              # → reports/baseline.json
CUDA_VISIBLE_DEVICES="" uv run --group train python src/train_lora.py --smoke   # teste rápido em CPU
```

Treino completo: `notebooks/kaggle_train.ipynb` numa GPU T4 do Kaggle.

## Estrutura

- `src/prepare_data.py`: limpeza, rótulo derivado da nota, dedup, split 80/10/10 estratificado (seed 42)
- `src/baseline.py`: baseline clássico
- `src/train_lora.py`: fine-tuning LoRA + avaliação no test + push opcional para o HF Hub
- `reports/*.json`: métricas versionadas (fonte dos números do README)
- `data/`, `outputs/`: não versionados

## Regras

- **O conjunto de teste é sagrado:** só para o número final. Escolha de hiperparâmetros e checkpoint usa a validação.
- Baseline e LLM devem ser avaliados no **mesmo** `data/test.parquet` gerado com seed 42.
- Números no README vêm de `reports/*.json`. Nunca escreva métrica que não foi produzida por um script.
- Ambiente local: Python 3.14, transformers 5.x (use `dtype=`, `warmup_steps` aceita float). A GPU local (GTX 1060) não funciona com o PyTorch atual: rode com `CUDA_VISIBLE_DEVICES=""`.
- O código deve continuar rodando no Kaggle (Python 3.11+; sem dependências exóticas).
- Comentários e docs em português, código e nomes em inglês, seguindo o estilo existente.
- Decisões técnicas novas → `DECISIONS.md`.

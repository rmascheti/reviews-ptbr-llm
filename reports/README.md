# Relatórios

Gerados na execução do Kaggle (commit `d9bad58`, 2× Tesla T4, transformers 5.16.1, peft 0.20.0).

- `baseline.json`: TF-IDF + Regressão Logística no test set.
- `lora.json`: Qwen2.5-0.5B + LoRA no test set (checkpoint do passo 1000, o de maior macro-F1 na validação).
- `training_curve.csv`: perda de treino a cada 50 passos e avaliações na validação (2.001 reviews) nos passos 500, 1000 e 1251, copiadas do log.

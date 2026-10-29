# Decisões técnicas

Formato: contexto → decisão → consequência. Quando uma decisão mudar, adicione uma entrada nova que substitui a anterior.

## T-001 · Dataset B2W-Reviews01

**Contexto:** precisávamos de reviews reais em português, públicos, com licença aberta e volume suficiente.
**Decisão:** B2W-Reviews01 (Americanas, ~132k reviews, CC-BY-4.0), baixado do GitHub oficial (a versão no HF usa script de carregamento, descontinuado no `datasets` atual).
**Consequência:** domínio é e-commerce, não delivery de comida. Citado como limitação.

## T-002 · Rótulo derivado da nota

**Contexto:** não há rótulo de sentimento anotado à mão.
**Decisão:** 1–2 = negativo, 3 = neutro, 4–5 = positivo (supervisão fraca).
**Consequência:** ruído de rótulo, principalmente no neutro; existe um teto de desempenho que nenhum modelo supera.

## T-003 · Deduplicação antes do split, split estratificado 80/10/10 com seed 42

**Contexto:** duplicatas entre splits inflam a métrica (vazamento); classes desbalanceadas (62/26/12).
**Decisão:** `drop_duplicates` no texto antes de dividir; `stratify` por rótulo; seed fixa.
**Consequência:** baseline e LLM comparáveis no mesmo test set; resultados reprodutíveis.

## T-004 · Macro-F1 como métrica principal

**Contexto:** prever sempre "positivo" já dá 62% de acurácia.
**Decisão:** macro-F1 para comparar modelos e escolher checkpoint; acurácia e F1 por classe como apoio.
**Consequência:** a classe neutra pesa tanto quanto as outras.

## T-005 · Baseline TF-IDF (1–2 gramas) + Regressão Logística balanceada

**Contexto:** um LLM só se justifica se superar algo barato.
**Decisão:** `sublinear_tf`, `min_df=3`, 200k features, `C=4.0`, `class_weight="balanced"`.
**Consequência:** macro-F1 0,753. `C` foi escolhido sem busca na validação: melhoria pendente para deixar a comparação mais justa.

## T-006 · Qwen2.5-0.5B base com cabeça de classificação + LoRA

**Contexto:** precisa caber numa T4 gratuita e ser barato para servir.
**Decisão:** `AutoModelForSequenceClassification` (pooling no último token não-pad), LoRA r=16, α=32, dropout 0,05, em q/k/v/o/gate/up/down. 8.800.896 parâmetros treináveis (1,75%).
**Consequência:** treino em fp16 na T4; adapter pequeno publicável no HF Hub. Versão *base* (não *instruct*) porque não há geração.

## T-007 · Hiperparâmetros de treino

**Decisão:** lr 2e-4, cosine com 5% de warmup, batch 16, max_len 256, 1 época sobre 40k exemplos estratificados, eval a cada 500 passos e melhor checkpoint por macro-F1 na validação (2k exemplos).
**Consequência:** treino estimado em 1–2 h na T4 (a confirmar). Sem busca de hiperparâmetros e com seed única: limitação documentada.

## T-008 · Treino no Kaggle; smoke test local em CPU

**Contexto:** a GTX 1060 (sm_61) não é suportada pelo PyTorch atual.
**Decisão:** `--smoke` (64 exemplos) roda local com `CUDA_VISIBLE_DEVICES=""`; treino real no notebook Kaggle.
**Consequência:** o notebook público no Kaggle também conta para o portfólio.

## T-009 · Remover torchao no Kaggle

**Contexto:** o Kaggle traz `torchao` 0.10.0. O `peft` 0.20 encontra o pacote instalado e exige versão ≥ 0.16, quebrando em `get_peft_model` mesmo sem quantização.
**Decisão:** `pip uninstall -y torchao` na primeira célula do notebook.
**Consequência:** nenhuma para o projeto (não usamos quantização). Se um dia usarmos, atualizar o `torchao` em vez de remover.

## T-010 · Comparar modelos no mesmo ambiente

**Contexto:** com os mesmos splits, o baseline deu macro-F1 0,7531 localmente (Python 3.14) e 0,7572 no Kaggle (Python 3.13), por diferenças de versão de bibliotecas.
**Decisão:** os números oficiais do README vêm da mesma execução no Kaggle que treina o LoRA, e o ambiente fica registrado no log.
**Consequência:** a diferença entre ambientes (~0,004) serve de referência: ganhos dessa ordem não são conclusivos.

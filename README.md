# Reviews PT-BR com LLM pequeno: quando o fine-tuning vale o custo?

Classificação de sentimento (negativo / neutro / positivo) em **~127 mil reviews reais de e-commerce em português** (B2W-Reviews01), comparando um baseline clássico com um LLM de 0,5B parâmetros ajustado com **LoRA**.

A pergunta de negócio: para uma empresa que processa milhões de avaliações, **quanto de qualidade um LLM fine-tunado entrega a mais, e a que custo de latência e infraestrutura?**

**Modelo publicado:** [huggingface.co/rmascheti/qwen2.5-0.5b-reviews-ptbr](https://huggingface.co/rmascheti/qwen2.5-0.5b-reviews-ptbr)

## Resultados

Test set de 12.689 reviews, os dois modelos avaliados na mesma execução ([`reports/`](reports/)).

| Modelo | Macro-F1 | Acurácia | Erros | Treino | Inferência |
|---|---|---|---|---|---|
| TF-IDF + Regressão Logística | 0,757 | 83,7% | 2.075 | 46 s (CPU) | 0,05 ms/review (CPU) |
| **Qwen2.5-0.5B + LoRA r=16** | **0,787** | **87,8%** | **1.547** | 46 min (2× T4) | 30 ms/review (2× T4) |

**O LLM erra 25% menos** (1.547 vs 2.075 reviews), com ganho de +0,029 no macro-F1. Como referência de ruído, o mesmo baseline variou 0,004 entre dois ambientes com versões diferentes de bibliotecas, então a diferença é bem maior que a variação de ambiente.

### F1 por classe

| Classe | Baseline (P / R / F1) | LoRA (P / R / F1) |
|---|---|---|
| negativo | 0,876 / 0,910 / 0,893 | 0,895 / 0,944 / **0,918** |
| neutro | 0,431 / 0,549 / 0,483 | 0,572 / 0,467 / **0,514** |
| positivo | 0,931 / 0,864 / 0,896 | 0,921 / 0,934 / **0,927** |

### O que os números dizem

- **O neutro continua sendo o gargalo.** A hipótese inicial era que o LLM ajudaria principalmente no neutro, onde o baseline mais erra. O ganho de F1 foi parecido nas três classes (+0,025 a +0,031), e o neutro segue perto de 0,51 para os dois modelos. Isso é coerente com o rótulo ruidoso: a nota 3 mistura textos positivos, negativos e mistos, e há um teto que nenhum modelo passa sem rótulos melhores.
- **Os modelos erram o neutro de jeitos diferentes.** O baseline usa `class_weight="balanced"` e chuta "neutro" com mais frequência (recall 0,55, precision 0,43). O LoRA, treinado sem pesos de classe, é mais conservador (precision 0,57, recall 0,47). Qual é melhor depende do uso: para triagem de reclamações ambíguas, recall importa mais.
- **Custo:** o LLM é cerca de 600× mais lento por review e precisa de GPU. Em 2× T4, a vazão foi de ~34 reviews/s, ou ~8 h para 1 milhão de reviews; o baseline faz o mesmo em menos de 1 minuto de CPU. Para um volume diário grande, uma arquitetura em cascata (o baseline decide quando está confiante e o LLM só vê os casos incertos) tende a capturar boa parte do ganho com uma fração do custo. Essa hipótese ainda não foi testada.

### Curva de treino

| Passo | Macro-F1 (val) | Acurácia (val) | Perda (val) |
|---|---|---|---|
| 500 | 0,645 | 85,6% | 0,795 |
| **1000** | **0,787** | 87,4% | 0,673 |
| 1251 (fim) | 0,779 | 87,6% | 0,656 |

O checkpoint publicado é o do passo 1000, escolhido pelo macro-F1 na validação (o test set não participou de nenhuma escolha). A perda de treino caiu de 7,6 nos primeiros 50 passos (cabeça de classificação aleatória) para ~1,2 no fim ([`reports/training_curve.csv`](reports/training_curve.csv)).

## Usar o modelo

```bash
uv sync --group train
uv run --group train python src/predict.py "Chegou rápido e bem embalado" "Bom, mas a entrega atrasou uma semana"
```

Saída com os exemplos padrão do script:

```
positivo  1.00  Produto excelente, chegou antes do prazo.
  neutro  0.88  Bom, mas a entrega atrasou uma semana.
negativo  0.99  Veio quebrado e ninguém responde.
```

## Abordagem

1. **Dados** (`src/prepare_data.py`): título + texto do review, sentimento derivado da nota (1–2 negativo, 3 neutro, 4–5 positivo), deduplicação antes do split, split estratificado 80/10/10 com seed fixa.
2. **Baseline** (`src/baseline.py`): TF-IDF de uni+bigramas, regressão logística com `class_weight="balanced"`.
3. **LLM** (`src/train_lora.py`): `Qwen2.5-0.5B` com cabeça de classificação, LoRA r=16 em todas as projeções de atenção e MLP (8,8M parâmetros treináveis, 1,75% do total), 40 mil exemplos, 1 época, batch efetivo 32 (8 por GPU × 2 GPUs × gradient accumulation 2), fp16, pico de 8,3 GB de memória por GPU.

As decisões e os problemas encontrados no caminho (memória da GPU, incompatibilidades de versão no Kaggle) estão em [`DECISIONS.md`](DECISIONS.md).

## Como reproduzir

```bash
uv sync
uv run python src/prepare_data.py   # baixa o CSV se não existir
uv run python src/baseline.py
uv run --group train python src/train_lora.py --smoke   # valida o pipeline em CPU
```

Treino completo: [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb) numa GPU T4 do Kaggle (~55 min com avaliação).

## Limitações

- **Rótulo derivado da nota**, não anotado à mão. Parte do erro no neutro vem do próprio dado.
- **Uma execução, uma seed.** Sem intervalo de confiança.
- **Sem busca de hiperparâmetros** no baseline (`C=4.0` fixo) nem no LoRA.
- **Latências em hardware diferente** (CPU vs GPU), medidas como vazão em lote, não como latência de uma requisição.
- **Domínio:** reviews de produtos de e-commerce, não de restaurantes ou entregas.

## Próximos passos

- [ ] Cascata baseline → LLM: medir quanto do ganho se mantém mandando só os casos incertos para o LLM
- [ ] Análise de erros: exemplos em que os modelos discordam e amostra rotulada à mão para medir o teto do neutro
- [ ] Demo no Hugging Face Spaces (Gradio)
- [ ] Destilação: LLM maior rotula o **motivo** da reclamação (entrega, defeito, produto errado, atendimento) e o modelo pequeno aprende

## Dados

[B2W-Reviews01](https://github.com/americanas-tech/b2w-reviews01) (Americanas S.A.), licença CC-BY-4.0. Real, Gabriel et al., *B2W-Reviews01: An open product reviews corpus*, STIL 2019.

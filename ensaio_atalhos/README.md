# Quando a IA aprende o atalho?

Ensaio computacional em Python sobre **aprendizado de atalhos** e **mudança de distribuição**. O notebook combina narrativa, código, tabelas e gráficos para investigar uma pergunta: um classificador com alta acurácia continua funcionando quando uma relação que aprendeu no treino deixa de valer?

## Como executar

Descompacte o ZIP e abra um terminal na pasta `ensaio_atalhos`. Use Python 3.12, versão usada na validação.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v
jupyter lab
```

No Windows PowerShell, substitua o comando de ativação por `.\.venv\Scripts\Activate.ps1`.

Abra `ensaio_computacional_ia.ipynb`, selecione o kernel do ambiente virtual e use **Restart Kernel and Run All Cells**. Mantenha o notebook e `experimento.py` na mesma pasta. A instalação requer acesso ao índice de pacotes; depois disso, os dados são gerados localmente, sem download.

| Arquivo | Função |
| --- | --- |
| `ensaio_computacional_ia.ipynb` | Ensaio principal, com resultados e gráficos já calculados |
| `experimento.py` | Geração dos dados, treino, simulação e resumo estatístico |
| `test_experimento.py` | Cinco testes automatizados, com `unittest` |
| `requirements.txt` | Bibliotecas da simulação e interface JupyterLab |
| `GUIA_APRESENTACAO.md` | Explicação curta e perguntas prováveis do professor |

## Desenho experimental

Os rótulos são −1 ou +1, com a mesma probabilidade. O **sinal estável** mantém a relação com o rótulo em todos os ambientes. O **atalho** é mais informativo no ambiente familiar, mas pode perder força ou inverter a associação no teste.

Comparamos três regressões logísticas, todas treinadas com 2.000 exemplos:

- **Familiar:** usa as duas características e só vê o ambiente +1.
- **Diversificado:** usa as duas características; metade dos exemplos vem de +1, metade de −1.
- **Só sinal estável:** usa os mesmos exemplos do treino familiar, mas não recebe a coluna do atalho.

Em cada uma das 80 repetições principais, os três modelos são avaliados nos mesmos exemplos de teste de cada ambiente, separados do treino. Os intervalos de confiança aproximados de 95% medem a incerteza da média entre repetições. Um segundo experimento avalia a mudança gradual do atalho com 40 repetições e 17 níveis.

## Resultados

| Estratégia | Teste familiar | Atalho invertido |
| --- | ---: | ---: |
| Treino familiar | 99,02% | 14,87% |
| Treino diversificado | 89,32% | 89,47% |
| Só sinal estável | 89,28% | 89,51% |

No teste invertido, a diversidade acrescentou **74,60 pontos percentuais** em média em relação ao treino familiar (IC aproximado de 95% da diferença pareada: **74,01 a 75,20 p.p.**). No teste familiar, custou 9,70 p.p. Isso ilustra uma troca: usar um atalho melhora a acurácia enquanto ele permanece válido, mas pode reduzir a robustez quando a associação muda.

O ensaio não testa dados reais nem demonstra generalização a qualquer ambiente: o ambiente invertido já faz parte do treino diversificado, embora seus exemplos de teste sejam novos. As referências do ensaio original foram mantidas no notebook.

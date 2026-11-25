# marvin-coletor-de-lixo

Robô coletor de lixo (Marvin) que aprende a tarefa por Aprendizado por Reforço, usando Q-Learning tabular com política ε-greedy, sem conhecer o modelo do ambiente.

Trabalho Prático III da disciplina Fundamentos de Inteligência Artificial (2026/2), Departamento de Engenharia Elétrica, UFMG.

## Status

> Atualizado em 25/11/2026

- [x] Leitura do enunciado e definição da estrutura do projeto
- [ ] Tarefa 1: modelagem do problema como MDP
- [ ] Tarefa 2: implementação do ambiente e do agente
- [ ] Tarefa 3: treinamento e curvas de aprendizado
- [ ] Tarefa 4: exploração versus explotação (varredura de ε e políticas de decaimento)
- [ ] Tarefa 5: análise da política aprendida
- [ ] Relatório em PDF (máximo de 6 páginas)
- [ ] Notebook final (.ipynb) comentado

## Sobre o problema

Marvin vive em uma grade 50 × 50 com paredes internas e lixo distribuído aleatoriamente. Ele percebe apenas a célula atual e as quatro vizinhas (norte, sul, leste, oeste), então o estado é a quíntupla de percepção, com 162 estados possíveis. Existem 7 ações (4 direções, aleatória, parado e coletar), e a Q-table tem 162 × 7 = 1134 entradas.

Recompensas: +10 ao coletar lixo, −1 ao tentar coletar onde não há lixo, −5 ao bater em parede ou na borda, 0 nos demais casos. Cada episódio tem 1000 passos e o treinamento usa 500 episódios.

## Estrutura do projeto

```
marvin-coletor-de-lixo/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/                      # Enunciado e material de apoio
├── src/                       # Código-fonte (módulos reutilizáveis)
│   ├── __init__.py
│   ├── configuracao.py        # Hiperparâmetros e constantes (α, γ, K, H, densidade, sementes)
│   ├── ambiente.py            # Classe Ambiente: grade, transição, recompensa, reset
│   ├── agente.py              # Classe AgenteQLearning: Q-table, ε-greedy, atualização
│   ├── agendas_epsilon.py     # Agendas de ε: constante, exponencial, harmônica
│   ├── treinamento.py         # Laço de treinamento e registro de métricas
│   ├── experimentos.py        # Varredura de ε e comparação de agendas (Tarefa 4)
│   ├── politica.py            # Extração e análise da política aprendida (Tarefa 5)
│   └── graficos.py            # Curvas de aprendizado, médias móveis, tabelas
├── scripts/                   # Pontos de entrada para rodar cada tarefa
│   ├── executar_treinamento.py
│   └── executar_experimentos.py
├── notebooks/                 # Notebook final entregável (.ipynb)
├── resultados/                # Saídas geradas (figuras, tabelas, arquivos .npy/.csv)
├── relatorio/                 # Fontes e PDF do relatório
└── testes/                    # Testes do ambiente e do agente
```

## Como executar

```bash
# Criar e ativar o ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# Instalar dependências
pip install -r requirements.txt
```

Os comandos de execução serão adicionados conforme os scripts forem implementados.

## Plano de trabalho

1. Modelar o MDP (estados, ações, recompensa, transição, γ, horizonte fixo).
2. Implementar o ambiente e testá-lo isoladamente (paredes, borda, coleta, ação aleatória).
3. Implementar o agente Q-Learning tabular com ε-greedy.
4. Treinar por 500 episódios, registrar as métricas por episódio e gerar as curvas com média móvel de 20 episódios.
5. Rodar os experimentos de exploração com α, γ, K e H fixos, agregando várias execuções com sementes diferentes.
6. Extrair e analisar a política aprendida com a melhor configuração.
7. Escrever o relatório e consolidar o notebook final.

- (preencher)
- (preencher)

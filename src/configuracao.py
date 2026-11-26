"""Constantes e hiperparâmetros do projeto Marvin.

Reúne os códigos das células, as ações, as recompensas e os hiperparâmetros
do treinamento, para que ambiente, agente e experimentos usem os mesmos
valores e a Tarefa 4 varie somente a estratégia de exploração.
"""

from dataclasses import dataclass
from enum import IntEnum


class Celula(IntEnum):
    """Códigos do conteúdo de uma célula da grade."""

    parede = -1
    vazio = 0
    lixo = 1


class Acao(IntEnum):
    """Ações disponíveis para Marvin, em ordem de índice na Q-table."""

    norte = 0
    sul = 1
    leste = 2
    oeste = 3
    aleatoria = 4
    parado = 5
    coletar = 6


# Deslocamento (delta_x, delta_y) de cada direção, com origem no canto superior esquerdo
deslocamento_por_acao = {
    Acao.norte: (0, -1),
    Acao.sul: (0, 1),
    Acao.leste: (1, 0),
    Acao.oeste: (-1, 0),
}

# Direções que podem ser sorteadas pela ação aleatória
acoes_de_movimento = (Acao.norte, Acao.sul, Acao.leste, Acao.oeste)

numero_de_acoes = len(Acao)

# Percepção: célula atual (vazio ou lixo) e quatro vizinhas (vazio, lixo ou parede)
numero_de_estados = 2 * 3**4


@dataclass(frozen=True)
class Recompensas:
    """Valores de recompensa definidos no enunciado."""

    coleta_com_sucesso: float = 10.0
    coleta_sem_lixo: float = -1.0
    movimento_invalido: float = -5.0
    demais_transicoes: float = 0.0


@dataclass(frozen=True)
class ConfiguracaoAmbiente:
    """Parâmetros do ambiente simulado.

    As densidades são premissas do grupo e devem ser declaradas no relatório.
    """

    tamanho_da_grade: int = 50
    densidade_de_lixo: float = 0.20
    densidade_de_paredes: float = 0.15
    passos_por_episodio: int = 1000
    posicao_inicial: tuple = (0, 0)


@dataclass(frozen=True)
class ConfiguracaoTreinamento:
    """Hiperparâmetros do Q-Learning.

    Valores iniciais propostos; devem ser congelados antes da Tarefa 4.
    """

    numero_de_episodios: int = 500
    taxa_de_aprendizado: float = 0.1
    fator_de_desconto: float = 0.9
    valor_inicial_q: float = 0.0
    janela_media_movel: int = 20
    semente_base: int = 42

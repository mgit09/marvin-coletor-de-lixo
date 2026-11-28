"""Extração e análise da política aprendida (Tarefa 5).

A política é a ação de maior valor Q em cada estado. Como cada execução do
treinamento gera uma Q-table própria, a política final é obtida por voto
majoritário entre as execuções, considerando só as execuções em que o estado
foi visitado. Estados nunca visitados ficam sem ação, pois o desempate do
argmax não carrega informação.
"""

import numpy as np
import pandas as pd

from src.ambiente import decodificar_percepcao
from src.configuracao import Acao, Celula, acoes_de_movimento, numero_de_acoes, numero_de_estados

NOME_DO_ESTADO_NAO_VISITADO = "não visitado"


def calcular_politica_majoritaria(q_tables, contagem_de_visitas):
    """Calcula a política por voto majoritário entre as execuções.

    Em cada execução, um estado visitado vota na sua ação de maior valor Q.
    Empates de voto são decididos pelo maior valor Q médio entre as execuções.

    Args:
        q_tables: Q-tables finais, formato (execuções, estados, ações).
        contagem_de_visitas: Visitas a cada par (s, a), mesmo formato de q_tables.

    Returns:
        Tupla (acoes, votos, q_medio). acoes tem um índice de ação por estado,
        ou -1 se o estado nunca foi visitado. votos tem formato (estados, ações)
        e q_medio é a média das Q-tables.
    """
    q_tables = np.asarray(q_tables)
    contagem_de_visitas = np.asarray(contagem_de_visitas)

    melhores_acoes = np.argmax(q_tables, axis=2)
    execucoes_com_visita, estados_com_visita = np.nonzero(contagem_de_visitas.sum(axis=2) > 0)

    votos = np.zeros((numero_de_estados, numero_de_acoes), dtype=np.int64)
    np.add.at(votos, (estados_com_visita, melhores_acoes[execucoes_com_visita, estados_com_visita]), 1)
    q_medio = q_tables.mean(axis=0)

    acoes = np.full(numero_de_estados, -1, dtype=np.int64)
    for estado in range(numero_de_estados):
        if votos[estado].sum() == 0:
            continue
        candidatas = np.flatnonzero(votos[estado] == votos[estado].max())
        acoes[estado] = candidatas[np.argmax(q_medio[estado, candidatas])]
    return acoes, votos, q_medio


def montar_tabela_da_politica(q_tables, contagem_de_visitas):
    """Monta a tabela estado -> ação com a percepção decodificada.

    Args:
        q_tables: Q-tables finais, formato (execuções, estados, ações).
        contagem_de_visitas: Visitas a cada par (s, a), mesmo formato de q_tables.

    Returns:
        DataFrame com uma linha por estado, com as colunas: estado, celula_atual,
        norte, sul, leste, oeste, numero_de_paredes, acao, execucoes_que_visitaram
        e concordancia (fração das execuções que votaram na ação escolhida).
    """
    acoes, votos, _ = calcular_politica_majoritaria(q_tables, contagem_de_visitas)

    linhas = []
    for estado in range(numero_de_estados):
        celula_atual, norte, sul, leste, oeste = decodificar_percepcao(estado)
        total_de_votos = int(votos[estado].sum())
        visitado = acoes[estado] >= 0
        linhas.append(
            {
                "estado": estado,
                "celula_atual": celula_atual.name,
                "norte": norte.name,
                "sul": sul.name,
                "leste": leste.name,
                "oeste": oeste.name,
                "numero_de_paredes": sum(vizinha == Celula.parede for vizinha in (norte, sul, leste, oeste)),
                "acao": Acao(acoes[estado]).name if visitado else NOME_DO_ESTADO_NAO_VISITADO,
                "execucoes_que_visitaram": total_de_votos,
                "concordancia": votos[estado, acoes[estado]] / total_de_votos if visitado else np.nan,
            }
        )
    return pd.DataFrame(linhas).set_index("estado")


def selecionar_estados_com_lixo(tabela):
    """Seleciona os estados visitados com lixo na célula atual (Tarefa 5a).

    Args:
        tabela: DataFrame de montar_tabela_da_politica.

    Returns:
        Subtabela com os estados em que celula_atual é lixo e que foram visitados.
    """
    visitados = tabela["acao"] != NOME_DO_ESTADO_NAO_VISITADO
    return tabela[(tabela["celula_atual"] == Celula.lixo.name) & visitados]


def selecionar_estados_cercados(tabela, minimo_de_paredes=1):
    """Seleciona os estados visitados cercados por paredes (Tarefa 5a).

    Args:
        tabela: DataFrame de montar_tabela_da_politica.
        minimo_de_paredes: Quantidade mínima de direções bloqueadas. O enunciado
            pede uma ou mais, e valores maiores deixam a tabela mais curta.

    Returns:
        Subtabela com os estados visitados com pelo menos essa quantidade de paredes.
    """
    visitados = tabela["acao"] != NOME_DO_ESTADO_NAO_VISITADO
    return tabela[(tabela["numero_de_paredes"] >= minimo_de_paredes) & visitados]


def resumir_politica(tabela):
    """Mede se a política aprendida tem o comportamento esperado.

    Considera somente estados visitados e calcula três frações:
    coletar quando há lixo na célula atual, andar em direção a um lixo vizinho
    quando não há lixo na célula atual, e não escolher uma direção bloqueada
    por parede quando a ação é um movimento.

    Args:
        tabela: DataFrame de montar_tabela_da_politica.

    Returns:
        Dicionário com, para cada medida, a fração e a quantidade de estados avaliados.
    """
    visitados = tabela[tabela["acao"] != NOME_DO_ESTADO_NAO_VISITADO]
    nomes_das_direcoes = [Acao(acao).name for acao in acoes_de_movimento]

    com_lixo_na_celula = visitados[visitados["celula_atual"] == Celula.lixo.name]
    coleta_em_lixo = (com_lixo_na_celula["acao"] == Acao.coletar.name).mean()

    sem_lixo_na_celula = visitados[visitados["celula_atual"] != Celula.lixo.name]
    com_lixo_vizinho = sem_lixo_na_celula[
        (sem_lixo_na_celula[nomes_das_direcoes] == Celula.lixo.name).any(axis=1)
    ]
    anda_para_o_lixo = [
        linha[linha["acao"]] == Celula.lixo.name if linha["acao"] in nomes_das_direcoes else False
        for _, linha in com_lixo_vizinho.iterrows()
    ]

    movimentos = visitados[visitados["acao"].isin(nomes_das_direcoes)]
    evita_parede = [linha[linha["acao"]] != Celula.parede.name for _, linha in movimentos.iterrows()]

    return {
        "coleta_quando_ha_lixo_na_celula": (float(coleta_em_lixo), len(com_lixo_na_celula)),
        "anda_para_lixo_vizinho": (float(np.mean(anda_para_o_lixo)) if anda_para_o_lixo else np.nan, len(anda_para_o_lixo)),
        "evita_direcao_com_parede": (float(np.mean(evita_parede)) if evita_parede else np.nan, len(evita_parede)),
    }


def tabela_da_politica_em_latex(tabela):
    """Gera o ambiente tabular em LaTeX (booktabs) com a política de uma subtabela.

    Args:
        tabela: Subtabela de montar_tabela_da_politica, por exemplo a de estados com lixo.

    Returns:
        Texto LaTeX com colunas da percepção e a ação escolhida.
    """
    linhas = [
        "\\begin{tabular}{llllll}",
        "\\toprule",
        "Atual & Norte & Sul & Leste & Oeste & Ação \\\\",
        "\\midrule",
    ]
    for _, linha in tabela.iterrows():
        celulas = [linha[coluna] for coluna in ("celula_atual", "norte", "sul", "leste", "oeste", "acao")]
        linhas.append(" & ".join(celulas) + " \\\\")
    linhas += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(linhas)

"""Testes da extração e análise da política."""

import numpy as np
import pytest

from src.ambiente import codificar_percepcao, decodificar_percepcao
from src.configuracao import Acao, Celula, acoes_de_movimento, numero_de_acoes, numero_de_estados
from src.politica import (
    NOME_DO_ESTADO_NAO_VISITADO,
    calcular_politica_majoritaria,
    montar_tabela_da_politica,
    resumir_politica,
    selecionar_estados_cercados,
    selecionar_estados_com_lixo,
    tabela_da_politica_em_latex,
)

ESTADO_COM_LIXO = codificar_percepcao(Celula.lixo, Celula.vazio, Celula.vazio, Celula.vazio, Celula.vazio)


def criar_q_tables(numero_de_execucoes=3):
    """Cria Q-tables e contagens zeradas.

    Args:
        numero_de_execucoes: Quantidade de execuções simuladas.

    Returns:
        Tupla (q_tables, contagem_de_visitas) com formato (execuções, 162, 7).
    """
    formato = (numero_de_execucoes, numero_de_estados, numero_de_acoes)
    return np.zeros(formato), np.zeros(formato, dtype=np.int64)


def test_voto_majoritario_escolhe_a_acao_mais_votada():
    """Duas execuções votam em coletar e uma em norte, então a política é coletar."""
    q_tables, visitas = criar_q_tables()
    visitas[:, ESTADO_COM_LIXO, :] = 1
    q_tables[0, ESTADO_COM_LIXO, Acao.coletar] = 5.0
    q_tables[1, ESTADO_COM_LIXO, Acao.coletar] = 4.0
    q_tables[2, ESTADO_COM_LIXO, Acao.norte] = 9.0

    acoes, votos, _ = calcular_politica_majoritaria(q_tables, visitas)

    assert acoes[ESTADO_COM_LIXO] == Acao.coletar
    assert votos[ESTADO_COM_LIXO, Acao.coletar] == 2
    assert votos[ESTADO_COM_LIXO, Acao.norte] == 1


def test_empate_de_votos_usa_o_maior_valor_q_medio():
    """Com um voto para cada ação, vence a de maior Q médio."""
    q_tables, visitas = criar_q_tables(numero_de_execucoes=2)
    visitas[:, ESTADO_COM_LIXO, :] = 1
    q_tables[0, ESTADO_COM_LIXO, Acao.coletar] = 3.0
    q_tables[1, ESTADO_COM_LIXO, Acao.sul] = 8.0

    acoes, _, _ = calcular_politica_majoritaria(q_tables, visitas)

    assert acoes[ESTADO_COM_LIXO] == Acao.sul


def test_estado_nunca_visitado_fica_sem_acao():
    """Sem visitas não há voto, e o estado é marcado como não visitado."""
    q_tables, visitas = criar_q_tables()

    acoes, _, _ = calcular_politica_majoritaria(q_tables, visitas)
    tabela = montar_tabela_da_politica(q_tables, visitas)

    assert np.all(acoes == -1)
    assert set(tabela["acao"]) == {NOME_DO_ESTADO_NAO_VISITADO}


def test_execucao_que_nao_visitou_o_estado_nao_vota():
    """Só as execuções com visita contam como voto."""
    q_tables, visitas = criar_q_tables()
    visitas[0, ESTADO_COM_LIXO, Acao.coletar] = 1
    q_tables[0, ESTADO_COM_LIXO, Acao.coletar] = 2.0
    q_tables[1, ESTADO_COM_LIXO, Acao.norte] = 7.0

    tabela = montar_tabela_da_politica(q_tables, visitas)

    assert tabela.loc[ESTADO_COM_LIXO, "acao"] == "coletar"
    assert tabela.loc[ESTADO_COM_LIXO, "execucoes_que_visitaram"] == 1
    assert tabela.loc[ESTADO_COM_LIXO, "concordancia"] == pytest.approx(1.0)


def test_tabela_tem_todos_os_estados_e_decodifica_a_percepcao():
    """A tabela cobre os 162 estados e mostra a percepção por extenso."""
    q_tables, visitas = criar_q_tables()
    tabela = montar_tabela_da_politica(q_tables, visitas)

    assert len(tabela) == numero_de_estados
    assert tabela.loc[ESTADO_COM_LIXO, "celula_atual"] == "lixo"
    assert tabela.loc[ESTADO_COM_LIXO, "numero_de_paredes"] == 0


def test_selecoes_de_estados():
    """Há 81 estados com lixo na célula atual, e a seleção de paredes respeita o mínimo."""
    q_tables, visitas = criar_q_tables()
    visitas[:] = 1
    tabela = montar_tabela_da_politica(q_tables, visitas)

    assert len(selecionar_estados_com_lixo(tabela)) == 81
    assert (selecionar_estados_cercados(tabela, minimo_de_paredes=3)["numero_de_paredes"] >= 3).all()
    assert len(selecionar_estados_cercados(tabela, 1)) > len(selecionar_estados_cercados(tabela, 3))


def test_resumo_de_uma_politica_ideal():
    """Uma política que coleta, vai ao lixo e evita parede atinge 100% nas três medidas."""
    q_tables, visitas = criar_q_tables(numero_de_execucoes=1)
    visitas[:] = 1

    for estado in range(numero_de_estados):
        atual, *vizinhas = decodificar_percepcao(estado)
        if atual == Celula.lixo:
            melhor = Acao.coletar
        elif Celula.lixo in vizinhas:
            melhor = acoes_de_movimento[vizinhas.index(Celula.lixo)]
        else:
            melhor = Acao.parado
        q_tables[0, estado, melhor] = 1.0

    resumo = resumir_politica(montar_tabela_da_politica(q_tables, visitas))

    assert resumo["coleta_quando_ha_lixo_na_celula"][0] == 1.0
    assert resumo["anda_para_lixo_vizinho"][0] == 1.0
    assert resumo["evita_direcao_com_parede"][0] == 1.0


def test_tabela_em_latex_tem_cabecalho_e_linhas():
    """O texto usa booktabs e uma linha por estado."""
    q_tables, visitas = criar_q_tables()
    visitas[:] = 1
    subtabela = selecionar_estados_com_lixo(montar_tabela_da_politica(q_tables, visitas))

    texto = tabela_da_politica_em_latex(subtabela)

    assert "\\toprule" in texto and "\\bottomrule" in texto
    assert texto.count("\\\\") == len(subtabela) + 1

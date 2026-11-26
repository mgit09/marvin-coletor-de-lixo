"""Testes do ambiente do robô coletor de lixo Marvin."""

import numpy as np
import pytest

from src.ambiente import (
    Ambiente,
    ResultadoPasso,
    codificar_percepcao,
    decodificar_percepcao,
)
from src.configuracao import Acao, Celula, ConfiguracaoAmbiente, numero_de_estados


def criar_ambiente_vazio(passos_por_episodio=1000):
    """Cria um ambiente sem paredes nem lixo, para testes controlados.

    Args:
        passos_por_episodio: Duração do episódio.

    Returns:
        Ambiente com a grade totalmente vazia e Marvin na posição inicial.
    """
    configuracao = ConfiguracaoAmbiente(
        densidade_de_lixo=0.0,
        densidade_de_paredes=0.0,
        passos_por_episodio=passos_por_episodio,
    )
    return Ambiente(configuracao, semente=0)


def test_codificacao_e_decodificacao_cobrem_todos_os_estados():
    """Cada estado possui um índice único e a decodificação é inversa."""
    percepcoes = set()
    for indice in range(numero_de_estados):
        percepcao = decodificar_percepcao(indice)
        assert codificar_percepcao(*percepcao) == indice
        percepcoes.add(percepcao)
    assert len(percepcoes) == numero_de_estados


def test_codificacao_rejeita_parede_na_celula_atual():
    """A célula atual nunca pode ser parede."""
    with pytest.raises(ValueError):
        codificar_percepcao(Celula.parede, 0, 0, 0, 0)


def test_reset_posiciona_marvin_na_origem_sem_parede():
    """O episódio sempre começa em (0, 0), que está livre de parede."""
    for semente in range(20):
        ambiente = Ambiente(semente=semente)
        estado = ambiente.reset()
        assert ambiente.posicao == (0, 0)
        assert ambiente.grade[0, 0] != Celula.parede
        assert 0 <= estado < numero_de_estados


def test_mesma_semente_gera_o_mesmo_ambiente():
    """A reprodutibilidade depende da semente."""
    primeiro = Ambiente(semente=7)
    segundo = Ambiente(semente=7)
    assert np.array_equal(primeiro.grade, segundo.grade)


def test_reset_mantem_paredes_e_redistribui_lixo():
    """Paredes são fixas e o lixo muda a cada episódio."""
    ambiente = Ambiente(semente=3)
    paredes_antes = ambiente.paredes.copy()
    grade_antes = ambiente.grade.copy()

    ambiente.reset()

    assert np.array_equal(ambiente.paredes, paredes_antes)
    assert np.array_equal(ambiente.grade == Celula.parede, paredes_antes)
    assert not np.array_equal(ambiente.grade, grade_antes)


def test_densidades_sao_proximas_das_configuradas():
    """As frações sorteadas ficam próximas dos valores da configuração."""
    ambiente = Ambiente(semente=1)
    configuracao = ambiente.configuracao
    total_de_celulas = configuracao.tamanho_da_grade**2

    fracao_de_paredes = np.sum(ambiente.paredes) / total_de_celulas
    celulas_livres = total_de_celulas - np.sum(ambiente.paredes)
    fracao_de_lixo = ambiente.quantidade_de_lixo_restante / celulas_livres

    assert abs(fracao_de_paredes - configuracao.densidade_de_paredes) < 0.03
    assert abs(fracao_de_lixo - configuracao.densidade_de_lixo) < 0.03


def test_movimento_contra_a_borda_e_invalido():
    """Sair da grade penaliza com -5 e mantém a posição."""
    ambiente = criar_ambiente_vazio()
    for acao in (Acao.norte, Acao.oeste):
        _, recompensa, _, resultado = ambiente.passo(acao)
        assert recompensa == -5.0
        assert resultado == ResultadoPasso.movimento_invalido
        assert ambiente.posicao == (0, 0)


def test_movimento_contra_parede_e_invalido():
    """Andar em direção a uma parede penaliza com -5 e mantém a posição."""
    ambiente = criar_ambiente_vazio()
    ambiente.grade[0, 1] = Celula.parede

    _, recompensa, _, resultado = ambiente.passo(Acao.leste)

    assert recompensa == -5.0
    assert resultado == ResultadoPasso.movimento_invalido
    assert ambiente.posicao == (0, 0)


def test_movimento_valido_nao_tem_recompensa():
    """Movimentos válidos e a ação parado rendem recompensa zero."""
    ambiente = criar_ambiente_vazio()

    _, recompensa, _, _ = ambiente.passo(Acao.sul)
    assert recompensa == 0.0
    assert ambiente.posicao == (0, 1)

    _, recompensa, _, _ = ambiente.passo(Acao.parado)
    assert recompensa == 0.0
    assert ambiente.posicao == (0, 1)


def test_coleta_com_sucesso_remove_o_lixo():
    """Coletar onde há lixo rende +10 e deixa a célula vazia."""
    ambiente = criar_ambiente_vazio()
    ambiente.grade[0, 0] = Celula.lixo

    _, recompensa, _, resultado = ambiente.passo(Acao.coletar)

    assert recompensa == 10.0
    assert resultado == ResultadoPasso.coleta_com_sucesso
    assert ambiente.grade[0, 0] == Celula.vazio


def test_coleta_sem_lixo_penaliza():
    """Coletar em célula vazia rende -1."""
    ambiente = criar_ambiente_vazio()

    _, recompensa, _, resultado = ambiente.passo(Acao.coletar)

    assert recompensa == -1.0
    assert resultado == ResultadoPasso.coleta_sem_lixo


def test_percepcao_reflete_vizinhanca():
    """O estado codifica a célula atual e as quatro vizinhas."""
    ambiente = criar_ambiente_vazio()
    ambiente.posicao = (5, 5)
    ambiente.grade[5, 5] = Celula.lixo
    ambiente.grade[4, 5] = Celula.parede
    ambiente.grade[6, 5] = Celula.lixo

    percepcao = decodificar_percepcao(ambiente.obter_estado())

    assert percepcao == (Celula.lixo, Celula.parede, Celula.lixo, Celula.vazio, Celula.vazio)


def test_borda_e_percebida_como_parede():
    """Fora da grade, o sensor lê parede."""
    ambiente = criar_ambiente_vazio()

    percepcao = decodificar_percepcao(ambiente.obter_estado())

    assert percepcao == (Celula.vazio, Celula.parede, Celula.vazio, Celula.vazio, Celula.parede)


def test_acao_aleatoria_move_uma_celula_no_interior_da_grade():
    """Longe das bordas, a ação aleatória sempre desloca uma célula."""
    ambiente = criar_ambiente_vazio()
    ambiente.posicao = (25, 25)

    for _ in range(20):
        coluna_antes, linha_antes = ambiente.posicao
        _, recompensa, _, _ = ambiente.passo(Acao.aleatoria)
        coluna_depois, linha_depois = ambiente.posicao

        distancia = abs(coluna_depois - coluna_antes) + abs(linha_depois - linha_antes)
        assert distancia == 1
        assert recompensa == 0.0


def test_acao_aleatoria_cobre_as_quatro_direcoes():
    """Todas as direções aparecem em muitos sorteios."""
    ambiente = criar_ambiente_vazio(passos_por_episodio=10_000)
    deslocamentos = set()

    for _ in range(200):
        ambiente.posicao = (25, 25)
        ambiente.passo(Acao.aleatoria)
        deslocamentos.add((ambiente.posicao[0] - 25, ambiente.posicao[1] - 25))

    assert deslocamentos == {(0, -1), (0, 1), (1, 0), (-1, 0)}


def test_episodio_termina_no_horizonte_e_exige_reset():
    """O último passo sinaliza o fim e novos passos exigem reset()."""
    ambiente = criar_ambiente_vazio(passos_por_episodio=5)

    for passo in range(5):
        _, _, terminado, _ = ambiente.passo(Acao.parado)
        assert terminado == (passo == 4)

    with pytest.raises(RuntimeError):
        ambiente.passo(Acao.parado)

    ambiente.reset()
    ambiente.passo(Acao.parado)

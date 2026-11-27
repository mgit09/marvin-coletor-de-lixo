"""Testes do agente Q-Learning do robô coletor de lixo Marvin."""

import numpy as np
import pytest

from src.agente import AgenteQLearning
from src.ambiente import Ambiente
from src.configuracao import (
    Acao,
    ConfiguracaoTreinamento,
    numero_de_acoes,
    numero_de_estados,
)


def test_q_table_inicia_em_q0_com_dimensoes_corretas():
    """A Q-table tem 162 x 7 entradas, todas iguais a Q0."""
    configuracao = ConfiguracaoTreinamento(valor_inicial_q=2.5)
    agente = AgenteQLearning(configuracao)

    assert agente.q_table.shape == (numero_de_estados, numero_de_acoes)
    assert np.all(agente.q_table == 2.5)
    assert agente.fracao_de_pares_visitados() == 0.0


def test_atualizacao_segue_a_equacao_2():
    """Q(s,a) = (1 - alpha) Q(s,a) + alpha [r + gamma max Q(s',a')]."""
    configuracao = ConfiguracaoTreinamento(taxa_de_aprendizado=0.1, fator_de_desconto=0.9)
    agente = AgenteQLearning(configuracao)
    agente.q_table[3, Acao.coletar] = 2.0
    agente.q_table[7, Acao.sul] = 5.0

    erro = agente.atualizar(estado=3, acao=Acao.coletar, recompensa=10.0, novo_estado=7)

    # Alvo = 10 + 0,9 * 5 = 14,5 e novo valor = 0,9 * 2 + 0,1 * 14,5 = 3,25
    assert agente.q_table[3, Acao.coletar] == pytest.approx(3.25)
    assert erro == pytest.approx(12.5)
    assert agente.contagem_de_visitas[3, Acao.coletar] == 1


def test_atualizacao_mantem_o_bootstrapping_sempre():
    """O valor futuro entra na atualização mesmo sem recompensa."""
    agente = AgenteQLearning(ConfiguracaoTreinamento(taxa_de_aprendizado=0.5, fator_de_desconto=0.9))
    agente.q_table[1, :] = 4.0

    agente.atualizar(estado=0, acao=Acao.norte, recompensa=0.0, novo_estado=1)

    assert agente.q_table[0, Acao.norte] == pytest.approx(0.5 * 0.9 * 4.0)


def test_epsilon_zero_explota_a_melhor_acao():
    """Sem exploração, o agente escolhe sempre a ação de maior valor."""
    agente = AgenteQLearning(epsilon=0.0, semente=0)
    agente.q_table[5, Acao.leste] = 1.0

    acoes = {agente.escolher_acao(5) for _ in range(200)}

    assert acoes == {Acao.leste}


def test_epsilon_um_explora_todas_as_acoes_uniformemente():
    """Com epsilon = 1, as 7 ações aparecem com frequência próxima de 1/7."""
    agente = AgenteQLearning(epsilon=1.0, semente=0)
    agente.q_table[5, Acao.leste] = 100.0

    contagem = np.zeros(numero_de_acoes)
    for _ in range(7000):
        contagem[agente.escolher_acao(5)] += 1

    assert np.all(np.abs(contagem / 7000 - 1 / numero_de_acoes) < 0.03)


def test_desempate_sorteia_entre_acoes_com_o_mesmo_valor():
    """Com a Q-table zerada, nenhuma ação é favorecida pelo índice."""
    agente = AgenteQLearning(epsilon=0.0, semente=0)

    acoes = {agente.escolher_acao(0) for _ in range(500)}

    assert len(acoes) == numero_de_acoes


def test_epsilon_invalido_e_rejeitado():
    """Epsilon fora de [0, 1] gera erro."""
    agente = AgenteQLearning()
    for valor in (-0.1, 1.1):
        with pytest.raises(ValueError):
            agente.definir_epsilon(valor)


def test_hiperparametros_invalidos_sao_rejeitados():
    """Alpha e gamma fora dos intervalos permitidos geram erro."""
    with pytest.raises(ValueError):
        AgenteQLearning(ConfiguracaoTreinamento(taxa_de_aprendizado=0.0))
    with pytest.raises(ValueError):
        AgenteQLearning(ConfiguracaoTreinamento(fator_de_desconto=1.5))


def test_fracao_visitada_conta_pares_distintos():
    """A fração conta cada par (s, a) uma única vez."""
    agente = AgenteQLearning()
    agente.atualizar(0, Acao.norte, 0.0, 0)
    agente.atualizar(0, Acao.norte, 0.0, 0)
    agente.atualizar(1, Acao.sul, 0.0, 0)

    assert agente.fracao_de_pares_visitados() == pytest.approx(2 / (numero_de_estados * numero_de_acoes))


def test_politica_aprendida_devolve_uma_acao_por_estado():
    """A política gulosa tem um valor por estado, dentro do conjunto de ações."""
    agente = AgenteQLearning()
    agente.q_table[10, Acao.parado] = 3.0

    politica = agente.politica_aprendida()

    assert politica.shape == (numero_de_estados,)
    assert politica[10] == Acao.parado


def test_treinamento_curto_aprende_a_coletar_lixo_na_celula_atual():
    """Depois de treinar, em estado com lixo na célula atual a política é coletar."""
    ambiente = Ambiente(semente=0)
    agente = AgenteQLearning(epsilon=0.1, semente=0)

    for _ in range(30):
        estado = ambiente.reset()
        terminado = False
        while not terminado:
            acao = agente.escolher_acao(estado)
            novo_estado, recompensa, terminado, _ = ambiente.passo(acao)
            agente.atualizar(estado, acao, recompensa, novo_estado)
            estado = novo_estado

    politica = agente.politica_aprendida()
    estados_com_lixo_visitados = [
        indice
        for indice in range(numero_de_estados)
        if indice >= 81 and agente.contagem_de_visitas[indice, Acao.coletar] > 0
    ]

    assert len(estados_com_lixo_visitados) > 0
    acertos = sum(politica[indice] == Acao.coletar for indice in estados_com_lixo_visitados)
    assert acertos / len(estados_com_lixo_visitados) > 0.9

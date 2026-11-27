"""Testes do laço de treinamento."""

import numpy as np
import pytest

from src.agendas_epsilon import criar_agenda_constante, criar_agenda_exponencial
from src.configuracao import ConfiguracaoAmbiente, ConfiguracaoTreinamento
from src.treinamento import treinar

PASSOS_POR_EPISODIO = 100
NUMERO_DE_EPISODIOS = 5


def treinar_curto(agenda, semente=0):
    """Executa um treinamento pequeno para os testes.

    Args:
        agenda: Agenda de epsilon.
        semente: Semente da execução.

    Returns:
        ResultadoTreinamento com poucos episódios e poucos passos.
    """
    return treinar(
        agenda,
        ConfiguracaoTreinamento(numero_de_episodios=NUMERO_DE_EPISODIOS),
        ConfiguracaoAmbiente(passos_por_episodio=PASSOS_POR_EPISODIO),
        semente=semente,
    )


def test_resultado_tem_um_valor_por_episodio():
    """Todas as séries têm o tamanho K."""
    resultado = treinar_curto(criar_agenda_constante(0.1))

    series = (
        resultado.recompensa_por_episodio,
        resultado.lixos_coletados_por_episodio,
        resultado.coletas_malsucedidas_por_episodio,
        resultado.movimentos_invalidos_por_episodio,
        resultado.epsilon_por_episodio,
        resultado.fracao_visitada_por_episodio,
    )
    assert all(len(serie) == NUMERO_DE_EPISODIOS for serie in series)


def test_recompensa_e_consistente_com_os_eventos_contados():
    """Como o resto vale zero, recompensa = 10 * coletas - coletas falhas - 5 * colisões."""
    resultado = treinar_curto(criar_agenda_constante(0.3))

    esperado = (
        10 * resultado.lixos_coletados_por_episodio
        - 1 * resultado.coletas_malsucedidas_por_episodio
        - 5 * resultado.movimentos_invalidos_por_episodio
    )
    assert np.array_equal(resultado.recompensa_por_episodio, esperado)


def test_eventos_nao_passam_do_numero_de_passos():
    """Cada passo gera no máximo um evento."""
    resultado = treinar_curto(criar_agenda_constante(1.0))

    eventos = (
        resultado.lixos_coletados_por_episodio
        + resultado.coletas_malsucedidas_por_episodio
        + resultado.movimentos_invalidos_por_episodio
    )
    assert np.all(eventos <= PASSOS_POR_EPISODIO)


def test_epsilon_registrado_segue_a_agenda():
    """O epsilon de cada episódio é o da agenda, com k começando em 1."""
    agenda = criar_agenda_exponencial()
    resultado = treinar_curto(agenda)

    esperado = [agenda(episodio) for episodio in range(1, NUMERO_DE_EPISODIOS + 1)]
    assert resultado.epsilon_por_episodio == pytest.approx(esperado)


def test_fracao_visitada_nunca_diminui():
    """Um par (s, a) visitado continua contando como visitado."""
    resultado = treinar_curto(criar_agenda_constante(0.5))

    assert np.all(np.diff(resultado.fracao_visitada_por_episodio) >= 0)
    assert 0.0 < resultado.fracao_visitada_final <= 1.0


def test_mesma_semente_reproduz_o_treinamento():
    """Execuções com a mesma semente são idênticas e com sementes diferentes não são."""
    primeiro = treinar_curto(criar_agenda_constante(0.1), semente=1)
    segundo = treinar_curto(criar_agenda_constante(0.1), semente=1)
    terceiro = treinar_curto(criar_agenda_constante(0.1), semente=2)

    assert np.array_equal(primeiro.recompensa_por_episodio, segundo.recompensa_por_episodio)
    assert not np.array_equal(primeiro.recompensa_por_episodio, terceiro.recompensa_por_episodio)


def test_metricas_resumidas_para_a_tabela_da_tarefa_4():
    """Média final, melhor episódio e fração visitada saem do resultado."""
    resultado = treinar_curto(criar_agenda_constante(0.1))

    assert resultado.recompensa_media_final(janela=2) == pytest.approx(
        np.mean(resultado.recompensa_por_episodio[-2:])
    )
    assert resultado.melhor_recompensa == np.max(resultado.recompensa_por_episodio)

"""Testes das agendas de decaimento do epsilon."""

import pytest

from src.agendas_epsilon import (
    criar_agenda_constante,
    criar_agenda_exponencial,
    criar_agenda_harmonica,
)


def test_agenda_constante_nao_muda():
    """A agenda constante devolve sempre o mesmo valor."""
    agenda = criar_agenda_constante(0.1)
    assert all(agenda(episodio) == 0.1 for episodio in (1, 50, 500))


def test_agenda_exponencial_segue_a_formula():
    """epsilon_k = max(epsilon_min, epsilon_0 * d^k)."""
    agenda = criar_agenda_exponencial(epsilon_inicial=1.0, decaimento=0.99, epsilon_minimo=0.01)

    assert agenda(1) == pytest.approx(0.99)
    assert agenda(100) == pytest.approx(0.99**100)
    assert agenda(500) == 0.01


def test_agenda_harmonica_segue_a_formula():
    """epsilon_k = max(epsilon_min, 1/k)."""
    agenda = criar_agenda_harmonica(epsilon_minimo=0.01)

    assert agenda(1) == 1.0
    assert agenda(4) == pytest.approx(0.25)
    assert agenda(500) == 0.01


def test_agendas_reproduzem_a_figura_5_do_enunciado():
    """A harmônica cai abaixo de 0,05 por volta do episódio 20 e a exponencial perto do 299."""
    harmonica = criar_agenda_harmonica()
    exponencial = criar_agenda_exponencial()

    assert harmonica(20) == pytest.approx(0.05)
    assert harmonica(21) < 0.05
    assert exponencial(298) > 0.05
    assert exponencial(299) < 0.05


def test_agendas_nunca_ficam_abaixo_do_minimo():
    """O epsilon mínimo é respeitado em todos os episódios."""
    agendas = (criar_agenda_exponencial(), criar_agenda_harmonica())
    for agenda in agendas:
        assert all(agenda(episodio) >= 0.01 for episodio in range(1, 501))


def test_parametros_invalidos_sao_rejeitados():
    """Valores fora dos intervalos permitidos geram erro."""
    with pytest.raises(ValueError):
        criar_agenda_constante(1.5)
    with pytest.raises(ValueError):
        criar_agenda_exponencial(epsilon_inicial=0.5, epsilon_minimo=0.9)
    with pytest.raises(ValueError):
        criar_agenda_exponencial(decaimento=0.0)
    with pytest.raises(ValueError):
        criar_agenda_harmonica(epsilon_minimo=-0.1)

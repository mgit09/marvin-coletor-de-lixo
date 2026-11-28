"""Testes da agregação de experimentos e das tabelas."""

import numpy as np
import pytest

from src.configuracao import ConfiguracaoAmbiente, ConfiguracaoTreinamento
from src.experimentos import (
    AgregadoExperimento,
    executar_configuracao,
    formatar_numero,
    montar_tabela_de_sensibilidade,
    montar_tabela_resumo,
    tabela_em_latex,
    analisar_sensibilidade,
    varredura_de_epsilon,
)

CONFIGURACAO_TREINAMENTO = ConfiguracaoTreinamento(numero_de_episodios=4)
CONFIGURACAO_AMBIENTE = ConfiguracaoAmbiente(passos_por_episodio=60)
ARGUMENTOS_RAPIDOS = {
    "numero_de_execucoes": 3,
    "configuracao_treinamento": CONFIGURACAO_TREINAMENTO,
    "configuracao_ambiente": CONFIGURACAO_AMBIENTE,
    "numero_de_processos": 1,
}


def criar_agregado_curto(epsilon=0.1, **extras):
    """Cria um agregado pequeno com agenda constante.

    Args:
        epsilon: Valor de epsilon da agenda constante.
        **extras: Argumentos que substituem os padrões rápidos.

    Returns:
        AgregadoExperimento com 3 execuções de 4 episódios.
    """
    argumentos = {**ARGUMENTOS_RAPIDOS, **extras}
    return executar_configuracao("ε = 0,1", "teste", "constante", {"epsilon": epsilon}, **argumentos)


def test_agregado_empilha_as_execucoes():
    """As matrizes têm formato (execuções, episódios) e as Q-tables (execuções, 162, 7)."""
    agregado = criar_agregado_curto()

    assert agregado.numero_de_execucoes == 3
    assert agregado.recompensa_por_episodio.shape == (3, 4)
    assert agregado.fracao_visitada_por_episodio.shape == (3, 4)
    assert agregado.q_tables.shape == (3, 162, 7)
    assert agregado.contagem_de_visitas.shape == (3, 162, 7)


def test_paralelo_e_serial_dao_o_mesmo_resultado():
    """Usar dois processos não altera os números, porque cada execução tem semente própria."""
    serial = criar_agregado_curto(numero_de_processos=1)
    paralelo = criar_agregado_curto(numero_de_processos=2)

    assert np.array_equal(serial.recompensa_por_episodio, paralelo.recompensa_por_episodio)
    assert np.array_equal(serial.q_tables, paralelo.q_tables)


def test_execucoes_usam_sementes_diferentes():
    """As execuções de um mesmo agregado não são idênticas."""
    agregado = criar_agregado_curto()

    assert not np.array_equal(agregado.recompensa_por_episodio[0], agregado.recompensa_por_episodio[1])


def test_salvar_e_carregar_preserva_os_dados(tmp_path):
    """O arquivo .npz devolve exatamente o que foi salvo."""
    agregado = criar_agregado_curto()
    caminho = tmp_path / "agregado.npz"

    agregado.salvar(caminho)
    carregado = AgregadoExperimento.carregar(caminho)

    assert carregado.nome == agregado.nome
    assert carregado.metadados == agregado.metadados
    assert np.array_equal(carregado.recompensa_por_episodio, agregado.recompensa_por_episodio)
    assert np.array_equal(carregado.q_tables, agregado.q_tables)


def test_cache_reaproveita_resultados_com_mesmos_parametros(tmp_path):
    """O segundo uso lê o arquivo, e parâmetros diferentes forçam novo treinamento."""
    primeiro = criar_agregado_curto(pasta_de_dados=tmp_path)
    caminho = tmp_path / "teste.npz"
    assert caminho.exists()

    # Adultera o arquivo para provar que a segunda chamada o lê em vez de treinar
    primeiro.recompensa_por_episodio[:] = -999
    primeiro.salvar(caminho)
    reaproveitado = criar_agregado_curto(pasta_de_dados=tmp_path)
    assert np.all(reaproveitado.recompensa_por_episodio == -999)

    # Com outro epsilon os metadados mudam, então o cache é ignorado
    refeito = criar_agregado_curto(epsilon=0.3, pasta_de_dados=tmp_path)
    assert not np.all(refeito.recompensa_por_episodio == -999)


def test_varredura_gera_uma_configuracao_por_epsilon():
    """O dicionário segue a ordem dos valores e usa a mesma semente em todos."""
    agregados = varredura_de_epsilon(valores_de_epsilon=(0.0, 0.5, 1.0), **ARGUMENTOS_RAPIDOS)

    assert list(agregados) == ["ε = 0", "ε = 0,5", "ε = 1"]
    assert all(agregado.numero_de_execucoes == 3 for agregado in agregados.values())


def test_tabela_resumo_calcula_media_e_desvio():
    """As colunas da tabela batem com as estatísticas das execuções."""
    agregado = criar_agregado_curto()
    tabela = montar_tabela_resumo({"ε = 0,1": agregado}, janela=2)

    medias = agregado.recompensa_por_episodio[:, -2:].mean(axis=1)
    linha = tabela.loc["ε = 0,1"]
    assert linha["recompensa_media_final"] == pytest.approx(medias.mean())
    assert linha["recompensa_media_final_desvio"] == pytest.approx(medias.std(ddof=1))
    assert linha["melhor_episodio"] == pytest.approx(agregado.recompensa_por_episodio.max(axis=1).mean())
    assert 0 < linha["porcentagem_visitada"] <= 100
    assert linha["numero_de_execucoes"] == 3


def test_tabela_em_latex_tem_o_formato_esperado():
    """A tabela usa booktabs, vírgula decimal e o epsilon em modo matemático."""
    agregado = criar_agregado_curto()
    texto = tabela_em_latex(montar_tabela_resumo({"ε = 0,05": agregado}))

    assert texto.startswith("\\begin{tabular}")
    assert "\\toprule" in texto and "\\bottomrule" in texto
    assert "$\\epsilon = 0{,}05$" in texto
    assert "\\pm" in texto
    assert texto.rstrip().endswith("\\end{tabular}")


def test_sensibilidade_cobre_a_grade_de_alfa_e_gama():
    """Há uma linha por par (alpha, gamma) com os valores corretos."""
    agregados = analisar_sensibilidade(
        taxas_de_aprendizado=(0.1, 0.3),
        fatores_de_desconto=(0.8, 0.9),
        **{**ARGUMENTOS_RAPIDOS, "numero_de_execucoes": 2},
    )
    tabela = montar_tabela_de_sensibilidade(agregados)

    pares = set(zip(tabela["taxa_de_aprendizado"], tabela["fator_de_desconto"]))
    assert pares == {(0.1, 0.8), (0.1, 0.9), (0.3, 0.8), (0.3, 0.9)}


def test_formatar_numero_usa_virgula():
    """O separador decimal é a vírgula."""
    assert formatar_numero(391.123, 1) == "391,1"
    assert formatar_numero(0.5, 2) == "0,50"

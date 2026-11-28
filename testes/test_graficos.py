"""Testes das funções de gráficos."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.configuracao import ConfiguracaoAmbiente, ConfiguracaoTreinamento  # noqa: E402
from src.experimentos import (  # noqa: E402
    analisar_sensibilidade,
    executar_configuracao,
    executar_sensibilidade_hiperparametros,
    montar_tabela_de_sensibilidade,
)
from src.graficos import (  # noqa: E402
    calcular_media_movel,
    plotar_comparacao_de_configuracoes,
    plotar_curva_de_aprendizado,
    plotar_mapa_de_sensibilidade,
    plotar_metricas_de_diagnostico,
    plotar_sensibilidade_hiperparametros,
)


@pytest.fixture(scope="module")
def agregado_curto():
    """Cria um agregado pequeno, compartilhado entre os testes do módulo.

    Returns:
        AgregadoExperimento com 2 execuções de 5 episódios.
    """
    return executar_configuracao(
        "Constante (ε = 0,1)",
        "teste_grafico",
        "constante",
        {"epsilon": 0.1},
        numero_de_execucoes=2,
        configuracao_treinamento=ConfiguracaoTreinamento(numero_de_episodios=5),
        configuracao_ambiente=ConfiguracaoAmbiente(passos_por_episodio=50),
        numero_de_processos=1,
    )


def test_media_movel_usa_janela_parcial_no_inicio():
    """As primeiras posições usam os valores disponíveis e as demais a janela completa."""
    resultado = calcular_media_movel([1, 2, 3, 4, 5], janela=3)

    assert resultado == pytest.approx([1.0, 1.5, 2.0, 3.0, 4.0])


def test_media_movel_preserva_o_tamanho_da_serie():
    """A saída tem um valor por episódio."""
    assert len(calcular_media_movel(np.arange(500), janela=20)) == 500


def test_curva_de_aprendizado_salva_o_arquivo(tmp_path, agregado_curto):
    """A figura é criada e gravada em PNG."""
    caminho = tmp_path / "pasta" / "curva.png"

    figura = plotar_curva_de_aprendizado(agregado_curto.recompensa_por_episodio, caminho_de_saida=caminho)

    assert caminho.exists() and caminho.stat().st_size > 0
    assert "n = 2 execuções" in figura.axes[0].get_title(loc="left")
    plt.close(figura)


def test_curva_de_aprendizado_aceita_uma_unica_execucao():
    """Com uma execução não há faixa de desvio e o título usa o singular."""
    figura = plotar_curva_de_aprendizado(np.arange(30.0))

    assert "n = 1 execução" in figura.axes[0].get_title(loc="left")
    plt.close(figura)


def test_metricas_de_diagnostico_tem_quatro_paineis(tmp_path, agregado_curto):
    """O gráfico de diagnóstico mostra quatro métricas."""
    figura = plotar_metricas_de_diagnostico(agregado_curto, caminho_de_saida=tmp_path / "diagnostico.png")

    assert len(figura.axes) == 4
    assert (tmp_path / "diagnostico.png").exists()
    plt.close(figura)


def test_comparacao_desenha_uma_curva_por_configuracao(agregado_curto):
    """Cada configuração vira uma linha com sua legenda."""
    agregados = {"A": agregado_curto, "B": agregado_curto}

    figura = plotar_comparacao_de_configuracoes(agregados)

    assert len(figura.axes[0].lines) == 2
    assert [texto.get_text() for texto in figura.axes[0].get_legend().get_texts()] == ["A", "B"]
    plt.close(figura)


def test_comparacao_rejeita_mais_configuracoes_que_cores(agregado_curto):
    """A paleta validada tem seis cores, e uma sétima série é recusada."""
    agregados = {str(indice): agregado_curto for indice in range(7)}

    with pytest.raises(ValueError):
        plotar_comparacao_de_configuracoes(agregados)


def test_mapa_de_sensibilidade_salva_o_arquivo(tmp_path):
    """O mapa de calor é gerado a partir da tabela de sensibilidade."""
    agregados = analisar_sensibilidade(
        taxas_de_aprendizado=(0.1, 0.3),
        fatores_de_desconto=(0.8, 0.9),
        numero_de_execucoes=2,
        configuracao_treinamento=ConfiguracaoTreinamento(numero_de_episodios=3),
        configuracao_ambiente=ConfiguracaoAmbiente(passos_por_episodio=40),
        numero_de_processos=1,
    )
    tabela = montar_tabela_de_sensibilidade(agregados)

    figura = plotar_mapa_de_sensibilidade(tabela, tmp_path / "mapa.png")

    assert (tmp_path / "mapa.png").exists()
    plt.close(figura)


def test_media_movel_rejeita_parametros_invalidos():
    """Janela não positiva ou série vazia geram ValueError."""
    with pytest.raises(ValueError):
        calcular_media_movel([1.0, 2.0], janela=0)
    with pytest.raises(ValueError):
        calcular_media_movel([], janela=5)


def test_painel_sensibilidade_hiperparametros_salva_arquivo(tmp_path):
    """O painel triplo de sensibilidade é gerado e salvo em PNG."""
    configuracao_referencia = ConfiguracaoTreinamento(numero_de_episodios=3)
    tabela, _ = executar_sensibilidade_hiperparametros(
        configuracao_referencia=configuracao_referencia,
        configuracao_ambiente=ConfiguracaoAmbiente(passos_por_episodio=40),
        taxas_de_aprendizado=(0.05, 0.1),
        fatores_de_desconto=(0.5, 0.9),
        valores_iniciais_q=(-5.0, 0.0),
        numero_de_execucoes=2,
        janela_final=2,
        numero_de_processos=1,
    )

    destino = tmp_path / "sensibilidade_painel.png"
    figura = plotar_sensibilidade_hiperparametros(
        tabela_resumo=tabela,
        configuracao_referencia=configuracao_referencia,
        passos_por_episodio=40,
        caminho_de_saida=destino,
    )

    assert len(figura.axes) == 3
    assert destino.exists() and destino.stat().st_size > 0
    plt.close(figura)

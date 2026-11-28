"""Gráficos do relatório: curvas de aprendizado, diagnósticos e comparações.

Todas as funções devolvem a figura e, se receberem um caminho, também a salvam
em PNG. As cores seguem uma paleta categórica validada para daltonismo, usada
sempre na mesma ordem, e as curvas ruidosas recebem uma média móvel.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

CORES_CATEGORICAS = ("#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17")
ESTILOS_DE_LINHA = ("-", "--", "-.", ":", (0, (5, 1)), (0, (1, 1)))
COR_PRINCIPAL = CORES_CATEGORICAS[0]
COR_DO_TEXTO = "#191919"
COR_DO_TEXTO_SECUNDARIO = "#666666"
COR_DA_GRADE = "#e7e7e7"
COR_DOS_EIXOS = "#c2c2c2"
RAMPA_SEQUENCIAL = ("#cce3fd", "#84b9f3", "#338ced", "#1b64be", "#0e3d7e")
TAMANHO_PADRAO = (7.0, 3.4)
JANELA_PADRAO = 20


def aplicar_estilo():
    """Aplica o estilo comum: eixos discretos, grade tênue e texto legível."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COR_DOS_EIXOS,
            "axes.labelcolor": COR_DO_TEXTO_SECUNDARIO,
            "axes.titlecolor": COR_DO_TEXTO,
            "axes.titlesize": 11,
            "axes.titleweight": "regular",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": COR_DA_GRADE,
            "grid.linewidth": 0.8,
            "xtick.color": COR_DO_TEXTO_SECUNDARIO,
            "ytick.color": COR_DO_TEXTO_SECUNDARIO,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 10,
            "savefig.facecolor": "white",
        }
    )


def calcular_media_movel(valores, janela=JANELA_PADRAO):
    """Calcula a média móvel causal, usando janelas menores no início da série.

    Args:
        valores: Série de valores, um por episódio.
        janela: Tamanho da janela, em episódios.

    Returns:
        Vetor do mesmo tamanho da série, em que cada posição é a média dos
        últimos valores disponíveis, no máximo igual ao tamanho da janela.
    """
    valores = np.asarray(valores, dtype=float)
    soma_acumulada = np.concatenate(([0.0], np.cumsum(valores)))
    fins = np.arange(1, len(valores) + 1)
    inicios = np.maximum(fins - janela, 0)
    return (soma_acumulada[fins] - soma_acumulada[inicios]) / (fins - inicios)


def _descrever_execucoes(numero_de_execucoes):
    """Monta o texto com o número de execuções agregadas.

    Args:
        numero_de_execucoes: Quantidade de execuções.

    Returns:
        Texto como "n = 10 execuções".
    """
    palavra = "execução" if numero_de_execucoes == 1 else "execuções"
    return f"n = {numero_de_execucoes} {palavra}"


def _finalizar(figura, caminho_de_saida):
    """Salva a figura em PNG quando um caminho é informado.

    Args:
        figura: Figura do matplotlib.
        caminho_de_saida: Caminho do arquivo ou None para não salvar.
    """
    if caminho_de_saida is not None:
        caminho_de_saida = Path(caminho_de_saida)
        caminho_de_saida.parent.mkdir(parents=True, exist_ok=True)
        figura.savefig(caminho_de_saida, dpi=200, bbox_inches="tight")


def _plotar_serie_com_media_movel(eixo, serie, janela, rotulo_da_serie, rotulo_da_media):
    """Desenha uma série em tom claro e sua média móvel em destaque.

    Args:
        eixo: Eixo do matplotlib.
        serie: Valores por episódio.
        janela: Janela da média móvel.
        rotulo_da_serie: Legenda da série bruta.
        rotulo_da_media: Legenda da média móvel.
    """
    episodios = np.arange(1, len(serie) + 1)
    eixo.plot(episodios, serie, color=COR_PRINCIPAL, alpha=0.4, linewidth=0.8, label=rotulo_da_serie)
    eixo.plot(
        episodios,
        calcular_media_movel(serie, janela),
        color=COR_PRINCIPAL,
        linewidth=2,
        label=rotulo_da_media,
    )


def plotar_curva_de_aprendizado(
    recompensa_por_execucao, titulo="Curva de aprendizado", janela=JANELA_PADRAO, caminho_de_saida=None
):
    """Plota a recompensa total por episódio com média móvel (Tarefa 3a).

    Com várias execuções, mostra a média entre elas e uma faixa de um desvio padrão.

    Args:
        recompensa_por_execucao: Matriz (execuções, episódios) ou vetor de uma execução.
        titulo: Título do gráfico.
        janela: Janela da média móvel.
        caminho_de_saida: Caminho do PNG, ou None para não salvar.

    Returns:
        Figura do matplotlib.
    """
    aplicar_estilo()
    dados = np.atleast_2d(np.asarray(recompensa_por_execucao, dtype=float))
    numero_de_execucoes, numero_de_episodios = dados.shape
    episodios = np.arange(1, numero_de_episodios + 1)
    media = dados.mean(axis=0)

    figura, eixo = plt.subplots(figsize=TAMANHO_PADRAO)
    if numero_de_execucoes > 1:
        desvio = dados.std(axis=0, ddof=1)
        eixo.fill_between(
            episodios,
            media - desvio,
            media + desvio,
            color=COR_PRINCIPAL,
            alpha=0.15,
            linewidth=0,
            label="±1 desvio padrão entre execuções",
        )
    rotulo_da_serie = "Média entre execuções" if numero_de_execucoes > 1 else "Recompensa do episódio"
    _plotar_serie_com_media_movel(eixo, media, janela, rotulo_da_serie, f"Média móvel ({janela} episódios)")

    eixo.set_title(f"{titulo} ({_descrever_execucoes(numero_de_execucoes)})", loc="left")
    eixo.set_xlabel("Episódio")
    eixo.set_ylabel("Recompensa total no episódio")
    eixo.legend(loc="lower right")
    _finalizar(figura, caminho_de_saida)
    return figura


def plotar_metricas_de_diagnostico(agregado, janela=JANELA_PADRAO, caminho_de_saida=None):
    """Plota lixos coletados, coletas malsucedidas, movimentos inválidos e Q-table visitada (Tarefa 3b e 3c).

    Args:
        agregado: AgregadoExperimento da configuração.
        janela: Janela da média móvel.
        caminho_de_saida: Caminho do PNG, ou None para não salvar.

    Returns:
        Figura do matplotlib.
    """
    aplicar_estilo()
    figura, eixos = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True)
    quantidade = agregado.numero_de_execucoes

    paineis = (
        (eixos[0, 0], agregado.lixos_coletados_por_episodio, "Lixos coletados por episódio"),
        (eixos[0, 1], agregado.coletas_malsucedidas_por_episodio, "Coletas malsucedidas por episódio"),
        (eixos[1, 0], agregado.movimentos_invalidos_por_episodio, "Movimentos inválidos por episódio"),
    )
    for eixo, matriz, titulo in paineis:
        _plotar_serie_com_media_movel(
            eixo, matriz.mean(axis=0), janela, "Média entre execuções", f"Média móvel ({janela} episódios)"
        )
        eixo.set_title(titulo, loc="left", fontsize=10)
        eixo.set_ylabel("Eventos")

    episodios = np.arange(1, agregado.fracao_visitada_por_episodio.shape[1] + 1)
    eixo_da_fracao = eixos[1, 1]
    eixo_da_fracao.plot(
        episodios, 100.0 * agregado.fracao_visitada_por_episodio.mean(axis=0), color=COR_PRINCIPAL, linewidth=2
    )
    eixo_da_fracao.set_title("Pares (s, a) visitados", loc="left", fontsize=10)
    eixo_da_fracao.set_ylabel("% da Q-table")

    eixos[1, 0].set_xlabel("Episódio")
    eixos[1, 1].set_xlabel("Episódio")
    eixos[0, 0].legend(loc="lower right", fontsize=8)
    figura.suptitle(f"{agregado.nome} ({_descrever_execucoes(quantidade)})", x=0.01, ha="left", fontsize=11)
    figura.tight_layout()
    _finalizar(figura, caminho_de_saida)
    return figura


def plotar_comparacao_de_configuracoes(
    agregados, titulo="Comparação de configurações", janela=JANELA_PADRAO, caminho_de_saida=None
):
    """Plota a média móvel da recompensa de várias configurações no mesmo gráfico.

    Cada configuração recebe uma cor da paleta, na ordem em que aparece, e um
    estilo de linha próprio, para que a leitura não dependa só da cor.

    Args:
        agregados: Dicionário nome -> AgregadoExperimento, com até seis configurações.
        titulo: Título do gráfico.
        janela: Janela da média móvel.
        caminho_de_saida: Caminho do PNG, ou None para não salvar.

    Returns:
        Figura do matplotlib.

    Raises:
        ValueError: Se houver mais configurações do que cores disponíveis.
    """
    if len(agregados) > len(CORES_CATEGORICAS):
        raise ValueError("Há mais configurações do que cores categóricas validadas")

    aplicar_estilo()
    figura, eixo = plt.subplots(figsize=TAMANHO_PADRAO)
    execucoes = {agregado.numero_de_execucoes for agregado in agregados.values()}

    for indice, (nome, agregado) in enumerate(agregados.items()):
        media = agregado.recompensa_por_episodio.mean(axis=0)
        episodios = np.arange(1, len(media) + 1)
        eixo.plot(
            episodios,
            calcular_media_movel(media, janela),
            color=CORES_CATEGORICAS[indice],
            linestyle=ESTILOS_DE_LINHA[indice],
            linewidth=1.8,
            label=nome,
        )

    descricao = _descrever_execucoes(execucoes.pop()) if len(execucoes) == 1 else "execuções variáveis"
    eixo.set_title(f"{titulo} (média móvel de {janela} episódios, {descricao})", loc="left", fontsize=10)
    eixo.set_xlabel("Episódio")
    eixo.set_ylabel("Recompensa total no episódio")
    eixo.legend(loc="lower right", ncol=2 if len(agregados) > 3 else 1)
    _finalizar(figura, caminho_de_saida)
    return figura


def plotar_mapa_de_sensibilidade(tabela_de_sensibilidade, caminho_de_saida=None):
    """Plota a recompensa média final para cada par (alpha, gamma) como mapa de calor.

    Args:
        tabela_de_sensibilidade: DataFrame de montar_tabela_de_sensibilidade.
        caminho_de_saida: Caminho do PNG, ou None para não salvar.

    Returns:
        Figura do matplotlib.
    """
    aplicar_estilo()
    medias = tabela_de_sensibilidade.pivot(
        index="taxa_de_aprendizado", columns="fator_de_desconto", values="recompensa_media_final"
    ).sort_index()
    medias = medias.reindex(sorted(medias.columns), axis=1)
    numero_de_execucoes = int(tabela_de_sensibilidade["numero_de_execucoes"].iloc[0])

    mapa_de_cores = LinearSegmentedColormap.from_list("sequencial_azul", RAMPA_SEQUENCIAL)
    figura, eixo = plt.subplots(figsize=(5.2, 3.6))
    eixo.grid(False)
    imagem = eixo.imshow(medias.values, cmap=mapa_de_cores, aspect="auto")

    eixo.set_xticks(range(len(medias.columns)), [f"{valor:g}".replace(".", ",") for valor in medias.columns])
    eixo.set_yticks(range(len(medias.index)), [f"{valor:g}".replace(".", ",") for valor in medias.index])
    eixo.set_xlabel("Fator de desconto γ")
    eixo.set_ylabel("Taxa de aprendizado α")

    limite_do_contraste = medias.values.min() + 0.6 * (medias.values.max() - medias.values.min())
    for linha in range(medias.shape[0]):
        for coluna in range(medias.shape[1]):
            valor = medias.values[linha, coluna]
            cor_do_texto = "white" if valor > limite_do_contraste else COR_DO_TEXTO
            eixo.text(
                coluna,
                linha,
                f"{valor:.0f}",
                ha="center",
                va="center",
                color=cor_do_texto,
                fontsize=10,
            )

    barra = figura.colorbar(imagem, ax=eixo)
    barra.set_label("Recompensa média (últimos 50 episódios)", color=COR_DO_TEXTO_SECUNDARIO)
    eixo.set_title(f"Sensibilidade a α e γ ({_descrever_execucoes(numero_de_execucoes)})", loc="left", fontsize=10)
    _finalizar(figura, caminho_de_saida)
    return figura

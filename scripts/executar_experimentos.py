"""Executa os experimentos das Tarefas 3 e 4 e salva dados, tabelas e figuras.

Uso, a partir da raiz do projeto:
    python -m scripts.executar_experimentos --etapa tudo --execucoes 10

Os resultados de cada configuração ficam em resultados/dados/. Se o script for
interrompido ou rodado de novo com os mesmos parâmetros, as configurações já
calculadas são carregadas em vez de treinadas outra vez.
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from src.experimentos import (  # noqa: E402
    comparar_agendas,
    montar_tabela_de_sensibilidade,
    montar_tabela_resumo,
    tabela_em_latex,
    analisar_sensibilidade,
    varredura_de_epsilon,
)
from src.graficos import (  # noqa: E402
    plotar_comparacao_de_configuracoes,
    plotar_curva_de_aprendizado,
    plotar_mapa_de_sensibilidade,
    plotar_metricas_de_diagnostico,
)

PASTA_DE_RESULTADOS = Path(__file__).resolve().parent.parent / "resultados"
PASTA_DE_DADOS = PASTA_DE_RESULTADOS / "dados"
PASTA_DE_FIGURAS = PASTA_DE_RESULTADOS / "figuras"
PASTA_DE_TABELAS = PASTA_DE_RESULTADOS / "tabelas"


def ler_argumentos():
    """Lê os argumentos da linha de comando.

    Returns:
        Namespace com etapa, execuções, execuções da sensibilidade e processos.
    """
    analisador = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    analisador.add_argument(
        "--etapa",
        choices=("sensibilidade", "agendas", "varredura", "tudo"),
        default="tudo",
        help="Etapa a executar",
    )
    analisador.add_argument("--execucoes", type=int, default=10, help="Execuções por configuração")
    analisador.add_argument(
        "--execucoes-sensibilidade", type=int, default=5, help="Execuções por configuração no teste de sensibilidade"
    )
    analisador.add_argument("--processos", type=int, default=None, help="Processos em paralelo")
    return analisador.parse_args()


def salvar_tabela(tabela, nome_do_arquivo, titulo_da_primeira_coluna=None):
    """Salva a tabela em CSV e, se pedido, em LaTeX.

    Args:
        tabela: DataFrame a salvar.
        nome_do_arquivo: Nome base, sem extensão.
        titulo_da_primeira_coluna: Se informado, gera também o .tex com esse cabeçalho.
    """
    PASTA_DE_TABELAS.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(PASTA_DE_TABELAS / f"{nome_do_arquivo}.csv")
    if titulo_da_primeira_coluna is not None:
        conteudo = tabela_em_latex(tabela, titulo_da_primeira_coluna)
        (PASTA_DE_TABELAS / f"{nome_do_arquivo}.tex").write_text(conteudo, encoding="utf-8")


def executar_sensibilidade(argumentos):
    """Executa o teste de sensibilidade de alpha e gamma.

    Args:
        argumentos: Argumentos lidos da linha de comando.
    """
    agregados = analisar_sensibilidade(
        numero_de_execucoes=argumentos.execucoes_sensibilidade,
        numero_de_processos=argumentos.processos,
        pasta_de_dados=PASTA_DE_DADOS,
    )
    tabela = montar_tabela_de_sensibilidade(agregados)
    salvar_tabela(tabela.set_index(["taxa_de_aprendizado", "fator_de_desconto"]), "sensibilidade")
    figura = plotar_mapa_de_sensibilidade(tabela, PASTA_DE_FIGURAS / "sensibilidade.png")
    plt.close(figura)
    print(tabela.round(1).to_string(index=False))


def executar_varredura(argumentos):
    """Executa a varredura de epsilon fixo (Tarefa 4a).

    Args:
        argumentos: Argumentos lidos da linha de comando.
    """
    agregados = varredura_de_epsilon(
        numero_de_execucoes=argumentos.execucoes,
        numero_de_processos=argumentos.processos,
        pasta_de_dados=PASTA_DE_DADOS,
    )
    tabela = montar_tabela_resumo(agregados)
    salvar_tabela(tabela, "varredura_de_epsilon", titulo_da_primeira_coluna="Configuração")
    figura = plotar_comparacao_de_configuracoes(
        agregados, "Varredura de ε fixo", caminho_de_saida=PASTA_DE_FIGURAS / "varredura_de_epsilon.png"
    )
    plt.close(figura)
    print(tabela.round(1).to_string())


def executar_agendas(argumentos):
    """Executa a comparação das agendas e gera as figuras da Tarefa 3.

    Args:
        argumentos: Argumentos lidos da linha de comando.
    """
    agregados = comparar_agendas(
        numero_de_execucoes=argumentos.execucoes,
        numero_de_processos=argumentos.processos,
        pasta_de_dados=PASTA_DE_DADOS,
    )
    tabela = montar_tabela_resumo(agregados)
    salvar_tabela(tabela, "comparacao_de_agendas", titulo_da_primeira_coluna="Agenda")
    figura = plotar_comparacao_de_configuracoes(
        agregados, "Agendas de decaimento de ε", caminho_de_saida=PASTA_DE_FIGURAS / "comparacao_de_agendas.png"
    )
    plt.close(figura)

    # A agenda exponencial serve de referência para as figuras da Tarefa 3
    exponencial = next(agregado for nome, agregado in agregados.items() if nome.startswith("Exponencial"))
    figura = plotar_curva_de_aprendizado(
        exponencial.recompensa_por_episodio,
        "Curva de aprendizado, agenda exponencial",
        caminho_de_saida=PASTA_DE_FIGURAS / "curva_de_aprendizado.png",
    )
    plt.close(figura)
    figura = plotar_metricas_de_diagnostico(
        exponencial, caminho_de_saida=PASTA_DE_FIGURAS / "metricas_de_diagnostico.png"
    )
    plt.close(figura)
    print(tabela.round(1).to_string())


def main():
    """Executa as etapas pedidas e informa o tempo gasto em cada uma."""
    argumentos = ler_argumentos()
    etapas = {
        "sensibilidade": executar_sensibilidade,
        "agendas": executar_agendas,
        "varredura": executar_varredura,
    }
    escolhidas = etapas if argumentos.etapa == "tudo" else {argumentos.etapa: etapas[argumentos.etapa]}

    for nome, funcao in escolhidas.items():
        inicio = time.time()
        print(f"\n=== Etapa: {nome} ===", flush=True)
        funcao(argumentos)
        print(f"Tempo da etapa {nome}: {time.time() - inicio:.0f} s", flush=True)


if __name__ == "__main__":
    main()

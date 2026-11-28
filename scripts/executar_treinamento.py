"""Treina Marvin com a configuração de referência e gera as figuras da Tarefa 3.

Uso, a partir da raiz do projeto:
    python -m scripts.executar_treinamento --agenda constante --execucoes 10

Os hiperparâmetros padrão vêm de src/configuracao.py. Depois do teste de
sensibilidade, informe os valores escolhidos em --taxa-de-aprendizado e
--fator-de-desconto, ou atualize o arquivo de configuração.
"""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

RAIZ_DO_PROJETO = Path(__file__).resolve().parent.parent
if str(RAIZ_DO_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_DO_PROJETO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from scripts.executar_experimentos import PASTA_DE_DADOS, PASTA_DE_FIGURAS, salvar_tabela  # noqa: E402
from src.configuracao import ConfiguracaoTreinamento  # noqa: E402
from src.experimentos import executar_configuracao, montar_tabela_resumo  # noqa: E402
from src.graficos import plotar_curva_de_aprendizado, plotar_metricas_de_diagnostico  # noqa: E402


def ler_argumentos():
    """Lê os argumentos da linha de comando.

    Returns:
        Namespace com agenda, epsilon, hiperparâmetros, execuções e processos.
    """
    padrao = ConfiguracaoTreinamento()
    analisador = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    analisador.add_argument("--agenda", choices=("constante", "exponencial", "harmonica"), default="constante")
    analisador.add_argument("--epsilon", type=float, default=0.1, help="Valor de ε da agenda constante")
    analisador.add_argument("--taxa-de-aprendizado", type=float, default=padrao.taxa_de_aprendizado)
    analisador.add_argument("--fator-de-desconto", type=float, default=padrao.fator_de_desconto)
    analisador.add_argument("--execucoes", type=int, default=10, help="Execuções com sementes diferentes")
    analisador.add_argument("--processos", type=int, default=None, help="Processos em paralelo")
    return analisador.parse_args()


def montar_parametros_da_agenda(argumentos):
    """Monta os parâmetros da agenda escolhida, com os valores do enunciado.

    Args:
        argumentos: Argumentos lidos da linha de comando.

    Returns:
        Dicionário de argumentos do construtor da agenda.
    """
    if argumentos.agenda == "constante":
        return {"epsilon": argumentos.epsilon}
    if argumentos.agenda == "exponencial":
        return {"epsilon_inicial": 1.0, "decaimento": 0.99, "epsilon_minimo": 0.01}
    return {"epsilon_minimo": 0.01}


def main():
    """Treina a configuração final, salva os dados e gera tabela e figuras."""
    argumentos = ler_argumentos()
    configuracao = replace(
        ConfiguracaoTreinamento(),
        taxa_de_aprendizado=argumentos.taxa_de_aprendizado,
        fator_de_desconto=argumentos.fator_de_desconto,
    )
    nome = f"Configuração final (agenda {argumentos.agenda})"

    agregado = executar_configuracao(
        nome,
        "treinamento_final",
        argumentos.agenda,
        montar_parametros_da_agenda(argumentos),
        numero_de_execucoes=argumentos.execucoes,
        configuracao_treinamento=configuracao,
        numero_de_processos=argumentos.processos,
        pasta_de_dados=PASTA_DE_DADOS,
    )

    figura = plotar_curva_de_aprendizado(
        agregado.recompensa_por_episodio,
        f"Curva de aprendizado, agenda {argumentos.agenda}",
        caminho_de_saida=PASTA_DE_FIGURAS / "curva_de_aprendizado_final.png",
    )
    plt.close(figura)
    figura = plotar_metricas_de_diagnostico(
        agregado, caminho_de_saida=PASTA_DE_FIGURAS / "metricas_de_diagnostico_final.png"
    )
    plt.close(figura)

    tabela = montar_tabela_resumo({nome: agregado})
    salvar_tabela(tabela, "treinamento_final")
    print(tabela.round(1).T.to_string())


if __name__ == "__main__":
    main()

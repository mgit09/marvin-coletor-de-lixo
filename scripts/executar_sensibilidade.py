"""Executa somente o teste de sensibilidade de alpha e gamma.

Uso, a partir da raiz do projeto:
    python -m scripts.executar_sensibilidade --execucoes-sensibilidade 5

Equivale a "python -m scripts.executar_experimentos --etapa sensibilidade".
Gera a tabela em resultados/tabelas/ e o mapa de calor em resultados/figuras/.
"""

import argparse

import matplotlib

matplotlib.use("Agg")

from scripts.executar_experimentos import executar_sensibilidade  # noqa: E402


def main():
    """Lê os argumentos e executa o teste de sensibilidade."""
    analisador = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    analisador.add_argument(
        "--execucoes-sensibilidade", type=int, default=5, help="Execuções por configuração"
    )
    analisador.add_argument("--processos", type=int, default=None, help="Processos em paralelo")
    executar_sensibilidade(analisador.parse_args())


if __name__ == "__main__":
    main()

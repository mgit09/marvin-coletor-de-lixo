"""Executa somente o estudo marginal de sensibilidade para alpha, gamma e Q0.

Uso, a partir da raiz do projeto:
    python -m scripts.executar_sensibilidade --execucoes-sensibilidade 10

Equivale a "python -m scripts.executar_experimentos --etapa sensibilidade".
Gera a tabela CSV em resultados/tabelas/ e o painel em resultados/figuras/.
"""

import argparse
from pathlib import Path
import sys

RAIZ_DO_PROJETO = Path(__file__).resolve().parent.parent
if str(RAIZ_DO_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_DO_PROJETO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from scripts.executar_experimentos import executar_sensibilidade  # noqa: E402


def main():
    """Lê os argumentos e executa o estudo de sensibilidade."""
    analisador = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    analisador.add_argument(
        "--execucoes-sensibilidade", type=int, default=10, help="Execuções por configuração"
    )
    analisador.add_argument("--processos", type=int, default=None, help="Processos em paralelo")
    executar_sensibilidade(analisador.parse_args())


if __name__ == "__main__":
    main()

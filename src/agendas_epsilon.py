"""Agendas de decaimento do epsilon usadas na Tarefa 4.

Cada função cria uma agenda, isto é, uma função que recebe o número do
episódio k (começando em 1) e devolve o epsilon daquele episódio.
"""

EPSILON_MINIMO_PADRAO = 0.01


def criar_agenda_constante(epsilon=0.1):
    """Cria a agenda constante, epsilon_k = epsilon, usada também na varredura de ε fixo.

    Args:
        epsilon: Valor fixo de epsilon, entre 0 e 1.

    Returns:
        Função que recebe o episódio k e devolve o epsilon correspondente.

    Raises:
        ValueError: Se epsilon estiver fora de [0, 1].
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("Epsilon deve estar em [0, 1]")

    def agenda(episodio):
        """Devolve o epsilon constante.

        Args:
            episodio: Número do episódio, começando em 1.

        Returns:
            Valor de epsilon.
        """
        return epsilon

    return agenda


def criar_agenda_exponencial(epsilon_inicial=1.0, decaimento=0.99, epsilon_minimo=EPSILON_MINIMO_PADRAO):
    """Cria a agenda exponencial, epsilon_k = max(epsilon_min, epsilon_0 * d^k).

    Args:
        epsilon_inicial: Valor de epsilon_0.
        decaimento: Fator d aplicado a cada episódio, entre 0 e 1.
        epsilon_minimo: Piso do epsilon.

    Returns:
        Função que recebe o episódio k e devolve o epsilon correspondente.

    Raises:
        ValueError: Se algum parâmetro estiver fora do intervalo válido.
    """
    if not 0.0 <= epsilon_minimo <= epsilon_inicial <= 1.0:
        raise ValueError("É preciso ter 0 <= epsilon_minimo <= epsilon_inicial <= 1")
    if not 0.0 < decaimento <= 1.0:
        raise ValueError("O decaimento deve estar em (0, 1]")

    def agenda(episodio):
        """Devolve o epsilon do episódio pelo decaimento exponencial.

        Args:
            episodio: Número do episódio, começando em 1.

        Returns:
            Valor de epsilon.
        """
        return max(epsilon_minimo, epsilon_inicial * decaimento**episodio)

    return agenda


def criar_agenda_harmonica(epsilon_minimo=EPSILON_MINIMO_PADRAO):
    """Cria a agenda harmônica, epsilon_k = max(epsilon_min, 1/k).

    Args:
        epsilon_minimo: Piso do epsilon.

    Returns:
        Função que recebe o episódio k e devolve o epsilon correspondente.

    Raises:
        ValueError: Se epsilon_minimo estiver fora de [0, 1].
    """
    if not 0.0 <= epsilon_minimo <= 1.0:
        raise ValueError("O epsilon mínimo deve estar em [0, 1]")

    def agenda(episodio):
        """Devolve o epsilon do episódio pelo decaimento harmônico.

        Args:
            episodio: Número do episódio, começando em 1.

        Returns:
            Valor de epsilon.
        """
        return max(epsilon_minimo, 1.0 / episodio)

    return agenda

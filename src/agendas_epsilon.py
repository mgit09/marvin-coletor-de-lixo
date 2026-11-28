"""Agendas de decaimento do epsilon usadas no treinamento e na Tarefa 4.

Cada classe e função de fábrica define uma agenda serializável que recebe o
número do episódio k (começando em 1) e devolve o epsilon daquele episódio.
"""

from dataclasses import dataclass

EPSILON_MINIMO_PADRAO = 0.01


@dataclass(frozen=True)
class AgendaConstante:
    """Agenda que mantém o valor de epsilon fixo em todos os episódios.

    Attributes:
        epsilon: Valor fixo de exploração, entre 0 e 1.
    """

    epsilon: float = 0.1

    def __post_init__(self):
        """Valida o intervalo do epsilon configurado.

        Raises:
            ValueError: Se epsilon estiver fora de [0, 1].
        """
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("Epsilon deve estar em [0, 1]")

    def __call__(self, episodio):
        """Devolve o epsilon constante para o episódio informado.

        Args:
            episodio: Número do episódio, começando em 1.

        Returns:
            Valor fixo de epsilon.
        """
        return float(self.epsilon)


@dataclass(frozen=True)
class AgendaExponencial:
    """Agenda com decaimento exponencial limitado por um piso mínimo.

    Attributes:
        epsilon_inicial: Valor inicial epsilon_0.
        decaimento: Fator multiplicativo d aplicado a cada episódio.
        epsilon_minimo: Limite inferior do epsilon.
    """

    epsilon_inicial: float = 1.0
    decaimento: float = 0.99
    epsilon_minimo: float = EPSILON_MINIMO_PADRAO

    def __post_init__(self):
        """Valida os parâmetros do decaimento exponencial.

        Raises:
            ValueError: Se algum parâmetro estiver fora do intervalo válido.
        """
        if not 0.0 <= self.epsilon_minimo <= self.epsilon_inicial <= 1.0:
            raise ValueError("É preciso ter 0 <= epsilon_minimo <= epsilon_inicial <= 1")
        if not 0.0 < self.decaimento <= 1.0:
            raise ValueError("O decaimento deve estar em (0, 1]")

    def __call__(self, episodio):
        """Devolve o epsilon do episódio pelo decaimento exponencial.

        Args:
            episodio: Número do episódio, começando em 1.

        Returns:
            Valor de epsilon calculado para o episódio.
        """
        return float(max(self.epsilon_minimo, self.epsilon_inicial * self.decaimento**episodio))


@dataclass(frozen=True)
class AgendaHarmonica:
    """Agenda com decaimento harmônico 1/k limitado por um piso mínimo.

    Attributes:
        epsilon_minimo: Limite inferior do epsilon.
    """

    epsilon_minimo: float = EPSILON_MINIMO_PADRAO

    def __post_init__(self):
        """Valida o piso mínimo do decaimento harmônico.

        Raises:
            ValueError: Se epsilon_minimo estiver fora de [0, 1].
        """
        if not 0.0 <= self.epsilon_minimo <= 1.0:
            raise ValueError("O epsilon mínimo deve estar em [0, 1]")

    def __call__(self, episodio):
        """Devolve o epsilon do episódio pelo decaimento harmônico.

        Args:
            episodio: Número do episódio, começando em 1.

        Returns:
            Valor de epsilon calculado para o episódio.
        """
        return float(max(self.epsilon_minimo, 1.0 / episodio))


def criar_agenda_constante(epsilon=0.1):
    """Cria a agenda constante, epsilon_k = epsilon, usada também na varredura de epsilon fixo.

    Args:
        epsilon: Valor fixo de epsilon, entre 0 e 1.

    Returns:
        Objeto chamável que recebe o episódio k e devolve o epsilon correspondente.

    Raises:
        ValueError: Se epsilon estiver fora de [0, 1].
    """
    return AgendaConstante(epsilon=epsilon)


def criar_agenda_exponencial(
    epsilon_inicial=1.0,
    decaimento=0.99,
    epsilon_minimo=EPSILON_MINIMO_PADRAO,
):
    """Cria a agenda exponencial, epsilon_k = max(epsilon_min, epsilon_0 * d^k).

    Args:
        epsilon_inicial: Valor de epsilon_0.
        decaimento: Fator d aplicado a cada episódio, entre 0 e 1.
        epsilon_minimo: Piso do epsilon.

    Returns:
        Objeto chamável que recebe o episódio k e devolve o epsilon correspondente.

    Raises:
        ValueError: Se algum parâmetro estiver fora do intervalo válido.
    """
    return AgendaExponencial(
        epsilon_inicial=epsilon_inicial,
        decaimento=decaimento,
        epsilon_minimo=epsilon_minimo,
    )


def criar_agenda_harmonica(epsilon_minimo=EPSILON_MINIMO_PADRAO):
    """Cria a agenda harmônica, epsilon_k = max(epsilon_min, 1/k).

    Args:
        epsilon_minimo: Piso do epsilon.

    Returns:
        Objeto chamável que recebe o episódio k e devolve o epsilon correspondente.

    Raises:
        ValueError: Se epsilon_minimo estiver fora de [0, 1].
    """
    return AgendaHarmonica(epsilon_minimo=epsilon_minimo)

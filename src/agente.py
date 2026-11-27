"""Agente Q-Learning tabular do robô coletor de lixo Marvin.

Mantém a Q-table (matriz de estados por ações), escolhe ações pela política
epsilon-greedy da Eq. (4) do enunciado e atualiza as estimativas pela Eq. (2).
O agente não conhece o modelo do ambiente e aprende somente com as tuplas
(s, a, r, s') que observa ao atuar.
"""

import numpy as np

from src.configuracao import Acao, ConfiguracaoTreinamento, numero_de_acoes, numero_de_estados


class AgenteQLearning:
    """Agente que aprende a Q-table por Q-Learning com política epsilon-greedy.

    Attributes:
        configuracao: Hiperparâmetros do treinamento.
        epsilon: Probabilidade atual de explorar, entre 0 e 1.
        q_table: Matriz [estado, ação] com as estimativas de Q.
        contagem_de_visitas: Matriz [estado, ação] com o número de atualizações.
        gerador: Gerador de números aleatórios com semente própria.
    """

    def __init__(self, configuracao=None, epsilon=0.1, semente=None):
        """Cria o agente com a Q-table inicializada em Q0.

        Args:
            configuracao: Hiperparâmetros do treinamento. Usa o padrão se for None.
            epsilon: Probabilidade inicial de explorar.
            semente: Semente do gerador aleatório, para reprodutibilidade.

        Raises:
            ValueError: Se algum hiperparâmetro for inválido.
        """
        self.configuracao = configuracao or ConfiguracaoTreinamento()
        self._validar_configuracao()

        self.gerador = np.random.default_rng(semente)
        self.q_table = np.full(
            (numero_de_estados, numero_de_acoes),
            self.configuracao.valor_inicial_q,
            dtype=np.float64,
        )
        self.contagem_de_visitas = np.zeros((numero_de_estados, numero_de_acoes), dtype=np.int64)
        self.epsilon = 0.0
        self.definir_epsilon(epsilon)

    def _validar_configuracao(self):
        """Confere se a taxa de aprendizado e o fator de desconto são válidos.

        Raises:
            ValueError: Se algum valor estiver fora do intervalo permitido.
        """
        if not 0.0 < self.configuracao.taxa_de_aprendizado <= 1.0:
            raise ValueError("A taxa de aprendizado deve estar em (0, 1]")
        if not 0.0 <= self.configuracao.fator_de_desconto <= 1.0:
            raise ValueError("O fator de desconto deve estar em [0, 1]")

    def definir_epsilon(self, epsilon):
        """Atualiza a probabilidade de explorar, por exemplo a cada episódio.

        Args:
            epsilon: Novo valor de epsilon, entre 0 e 1.

        Raises:
            ValueError: Se epsilon estiver fora de [0, 1].
        """
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("Epsilon deve estar em [0, 1]")
        self.epsilon = float(epsilon)

    def _melhor_acao(self, estado):
        """Escolhe a ação de maior valor Q, sorteando entre as empatadas.

        O desempate aleatório evita que o agente prefira sempre a primeira
        ação enquanto a Q-table ainda tem valores iguais.

        Args:
            estado: Índice do estado percebido.

        Returns:
            Índice da ação de maior valor estimado.
        """
        valores = self.q_table[estado]
        candidatas = np.flatnonzero(valores == valores.max())
        return int(self.gerador.choice(candidatas))

    def escolher_acao(self, estado):
        """Escolhe uma ação pela política epsilon-greedy (Eq. 4).

        Com probabilidade epsilon explora, sorteando uniformemente entre as 7
        ações. Caso contrário, explota a ação de maior valor Q estimado.

        Args:
            estado: Índice do estado percebido.

        Returns:
            Ação escolhida.
        """
        if self.gerador.random() < self.epsilon:
            return Acao(int(self.gerador.integers(numero_de_acoes)))
        return Acao(self._melhor_acao(estado))

    def atualizar(self, estado, acao, recompensa, novo_estado):
        """Atualiza a Q-table com uma transição observada (Eq. 2).

        O termo de bootstrapping é sempre mantido, inclusive no último passo do
        episódio, porque o fim por limite de tempo é apenas um truncamento.

        Args:
            estado: Estado em que o agente estava.
            acao: Ação executada.
            recompensa: Recompensa recebida.
            novo_estado: Estado em que o agente chegou.

        Returns:
            Erro de diferença temporal (alvo menos predição), útil para diagnóstico.
        """
        taxa_de_aprendizado = self.configuracao.taxa_de_aprendizado
        fator_de_desconto = self.configuracao.fator_de_desconto

        predicao = self.q_table[estado, acao]
        alvo = recompensa + fator_de_desconto * self.q_table[novo_estado].max()

        self.q_table[estado, acao] = (1 - taxa_de_aprendizado) * predicao + taxa_de_aprendizado * alvo
        self.contagem_de_visitas[estado, acao] += 1
        return alvo - predicao

    def fracao_de_pares_visitados(self):
        """Calcula a fração de pares (s, a) atualizados ao menos uma vez (Tarefa 3c).

        Returns:
            Valor entre 0 e 1.
        """
        return float(np.mean(self.contagem_de_visitas > 0))

    def politica_aprendida(self):
        """Extrai a política gulosa da Q-table (Tarefa 5).

        Em estados com valores empatados, como os nunca visitados, o desempate
        escolhe a ação de menor índice, então esses estados devem ser
        identificados pela contagem de visitas antes de serem interpretados.

        Returns:
            Vetor com a ação de maior valor Q para cada estado.
        """
        return np.argmax(self.q_table, axis=1)

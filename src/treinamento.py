"""Laço de treinamento do Q-Learning tabular e registro das métricas.

Executa K episódios de H passos, seguindo o algoritmo do enunciado, e guarda
por episódio a recompensa total, os lixos coletados, as coletas malsucedidas,
os movimentos inválidos, o epsilon usado e a fração da Q-table visitada.
"""

from dataclasses import dataclass

import numpy as np

from src.agente import AgenteQLearning
from src.ambiente import Ambiente, ResultadoPasso
from src.configuracao import ConfiguracaoAmbiente, ConfiguracaoTreinamento


@dataclass
class ResultadoTreinamento:
    """Métricas de uma execução de treinamento, com um valor por episódio.

    Attributes:
        recompensa_por_episodio: Recompensa total acumulada em cada episódio.
        lixos_coletados_por_episodio: Coletas bem-sucedidas em cada episódio.
        coletas_malsucedidas_por_episodio: Coletas em célula sem lixo.
        movimentos_invalidos_por_episodio: Colisões com parede ou borda.
        epsilon_por_episodio: Epsilon usado em cada episódio.
        fracao_visitada_por_episodio: Fração de pares (s, a) visitados ao final de cada episódio.
        agente: Agente ao final do treinamento, com a Q-table aprendida.
    """

    recompensa_por_episodio: np.ndarray
    lixos_coletados_por_episodio: np.ndarray
    coletas_malsucedidas_por_episodio: np.ndarray
    movimentos_invalidos_por_episodio: np.ndarray
    epsilon_por_episodio: np.ndarray
    fracao_visitada_por_episodio: np.ndarray
    agente: AgenteQLearning

    def recompensa_media_final(self, janela=50):
        """Calcula a recompensa média dos últimos episódios (coluna da tabela da Tarefa 4a).

        Args:
            janela: Quantidade de episódios finais considerados.

        Returns:
            Média da recompensa total nos últimos episódios.
        """
        return float(np.mean(self.recompensa_por_episodio[-janela:]))

    @property
    def melhor_recompensa(self):
        """Maior recompensa total obtida em um único episódio."""
        return float(np.max(self.recompensa_por_episodio))

    @property
    def fracao_visitada_final(self):
        """Fração de pares (s, a) visitados ao final do treinamento (Tarefa 3c)."""
        return float(self.fracao_visitada_por_episodio[-1])


def treinar(
    agenda_de_epsilon,
    configuracao_treinamento=None,
    configuracao_ambiente=None,
    semente=None,
    mostrar_progresso=False,
):
    """Treina Marvin por K episódios de H passos com Q-Learning tabular.

    A semente é dividida em duas sequências independentes, uma para o ambiente
    e outra para o agente. Com a mesma semente, configurações diferentes de
    exploração enfrentam o mesmo mapa, o que permite comparações justas.

    Args:
        agenda_de_epsilon: Função que recebe o episódio k (a partir de 1) e devolve o epsilon.
        configuracao_treinamento: Hiperparâmetros. Usa o padrão se for None.
        configuracao_ambiente: Parâmetros do ambiente. Usa o padrão se for None.
        semente: Semente da execução. Usa a semente base da configuração se for None.
        mostrar_progresso: Se verdadeiro, exibe uma barra de progresso (requer tqdm).

    Returns:
        ResultadoTreinamento com as métricas por episódio e o agente treinado.
    """
    configuracao_treinamento = configuracao_treinamento or ConfiguracaoTreinamento()
    configuracao_ambiente = configuracao_ambiente or ConfiguracaoAmbiente()
    if semente is None:
        semente = configuracao_treinamento.semente_base

    semente_ambiente, semente_agente = np.random.SeedSequence(semente).spawn(2)
    ambiente = Ambiente(configuracao_ambiente, semente=semente_ambiente)
    agente = AgenteQLearning(configuracao_treinamento, semente=semente_agente)

    numero_de_episodios = configuracao_treinamento.numero_de_episodios
    recompensa_por_episodio = np.zeros(numero_de_episodios)
    lixos_coletados_por_episodio = np.zeros(numero_de_episodios, dtype=np.int64)
    coletas_malsucedidas_por_episodio = np.zeros(numero_de_episodios, dtype=np.int64)
    movimentos_invalidos_por_episodio = np.zeros(numero_de_episodios, dtype=np.int64)
    epsilon_por_episodio = np.zeros(numero_de_episodios)
    fracao_visitada_por_episodio = np.zeros(numero_de_episodios)

    episodios = range(numero_de_episodios)
    if mostrar_progresso:
        from tqdm import tqdm

        episodios = tqdm(episodios, desc="Treinamento", unit="episódio")

    for indice_do_episodio in episodios:
        epsilon = agenda_de_epsilon(indice_do_episodio + 1)
        agente.definir_epsilon(epsilon)
        epsilon_por_episodio[indice_do_episodio] = epsilon

        estado = ambiente.reset()
        terminado = False
        while not terminado:
            acao = agente.escolher_acao(estado)
            novo_estado, recompensa, terminado, resultado = ambiente.passo(acao)
            agente.atualizar(estado, acao, recompensa, novo_estado)
            estado = novo_estado

            recompensa_por_episodio[indice_do_episodio] += recompensa
            if resultado == ResultadoPasso.coleta_com_sucesso:
                lixos_coletados_por_episodio[indice_do_episodio] += 1
            elif resultado == ResultadoPasso.coleta_sem_lixo:
                coletas_malsucedidas_por_episodio[indice_do_episodio] += 1
            elif resultado == ResultadoPasso.movimento_invalido:
                movimentos_invalidos_por_episodio[indice_do_episodio] += 1

        fracao_visitada_por_episodio[indice_do_episodio] = agente.fracao_de_pares_visitados()

    return ResultadoTreinamento(
        recompensa_por_episodio=recompensa_por_episodio,
        lixos_coletados_por_episodio=lixos_coletados_por_episodio,
        coletas_malsucedidas_por_episodio=coletas_malsucedidas_por_episodio,
        movimentos_invalidos_por_episodio=movimentos_invalidos_por_episodio,
        epsilon_por_episodio=epsilon_por_episodio,
        fracao_visitada_por_episodio=fracao_visitada_por_episodio,
        agente=agente,
    )

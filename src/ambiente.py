"""Ambiente simulado do robô coletor de lixo Marvin.

Implementa a grade 50 x 50, a percepção local de cinco células, a função de
transição, a função de recompensa e o reinício de episódio descritos no
enunciado. O agente interage com o ambiente somente por meio de reset() e
passo(), sem acesso à grade, à posição ou às probabilidades de transição.
"""

from enum import IntEnum

import numpy as np

from src.configuracao import (
    Acao,
    Celula,
    ConfiguracaoAmbiente,
    Recompensas,
    acoes_de_movimento,
    deslocamento_por_acao,
    numero_de_estados,
)


class ResultadoPasso(IntEnum):
    """Classificação do que aconteceu em um passo, usada para registrar métricas."""

    coleta_com_sucesso = 0
    coleta_sem_lixo = 1
    movimento_invalido = 2
    outro = 3


def codificar_percepcao(celula_atual, norte, sul, leste, oeste):
    """Converte a quíntupla de percepção em um índice de estado.

    A célula atual ocupa o dígito mais significativo (base 2, pois nunca é
    parede) e as quatro vizinhas ocupam dígitos de base 3, na ordem norte,
    sul, leste e oeste, com parede = 0, vazio = 1 e lixo = 2.

    Args:
        celula_atual: Conteúdo da célula ocupada (vazio ou lixo).
        norte: Conteúdo da célula ao norte.
        sul: Conteúdo da célula ao sul.
        leste: Conteúdo da célula a leste.
        oeste: Conteúdo da célula a oeste.

    Returns:
        Índice do estado, entre 0 e numero_de_estados - 1.

    Raises:
        ValueError: Se a célula atual for parede.
    """
    if int(celula_atual) not in (Celula.vazio, Celula.lixo):
        raise ValueError("A célula atual não pode ser parede")

    indice = int(celula_atual)
    for vizinha in (norte, sul, leste, oeste):
        indice = indice * 3 + (int(vizinha) + 1)
    return indice


def decodificar_percepcao(indice_do_estado):
    """Converte um índice de estado de volta na quíntupla de percepção.

    Operação inversa de codificar_percepcao(), útil para analisar a política
    aprendida (Tarefa 5).

    Args:
        indice_do_estado: Índice do estado, entre 0 e numero_de_estados - 1.

    Returns:
        Tupla (celula_atual, norte, sul, leste, oeste) com valores de Celula.

    Raises:
        ValueError: Se o índice estiver fora do intervalo válido.
    """
    if not 0 <= indice_do_estado < numero_de_estados:
        raise ValueError("Índice de estado fora do intervalo válido")

    resto = int(indice_do_estado)
    vizinhas = []
    for _ in range(4):
        resto, digito = divmod(resto, 3)
        vizinhas.append(Celula(digito - 1))
    vizinhas.reverse()
    return (Celula(resto), *vizinhas)


class Ambiente:
    """Grade bidimensional em que Marvin coleta lixo.

    A disposição das paredes é sorteada uma única vez, na criação do ambiente,
    e permanece fixa. O lixo é sorteado novamente a cada reset(). A densidade
    de paredes é a fração das células da grade, e a densidade de lixo é a
    fração das células livres.

    Attributes:
        configuracao: Parâmetros do ambiente.
        recompensas: Valores de recompensa.
        gerador: Gerador de números aleatórios com semente própria.
        paredes: Matriz booleana [y, x] indicando as paredes.
        grade: Matriz [y, x] com os códigos de Celula da grade atual.
        posicao: Posição (x, y) atual de Marvin.
        passo_atual: Quantidade de passos executados no episódio.
    """

    def __init__(self, configuracao=None, recompensas=None, semente=None):
        """Cria o ambiente, sorteia as paredes e inicia o primeiro episódio.

        Args:
            configuracao: Parâmetros do ambiente. Usa o padrão se for None.
            recompensas: Valores de recompensa. Usa o padrão se for None.
            semente: Semente do gerador aleatório, para reprodutibilidade.
        """
        self.configuracao = configuracao or ConfiguracaoAmbiente()
        self.recompensas = recompensas or Recompensas()
        self._validar_configuracao()

        self.gerador = np.random.default_rng(semente)
        self.paredes = self._gerar_paredes()
        self.grade = None
        self.posicao = None
        self.passo_atual = 0
        self.reset()

    def _validar_configuracao(self):
        """Confere se os parâmetros do ambiente são coerentes.

        Raises:
            ValueError: Se algum parâmetro for inválido.
        """
        tamanho = self.configuracao.tamanho_da_grade
        coluna, linha = self.configuracao.posicao_inicial

        if tamanho <= 0:
            raise ValueError("O tamanho da grade deve ser positivo")
        if not (0 <= coluna < tamanho and 0 <= linha < tamanho):
            raise ValueError("A posição inicial deve estar dentro da grade")
        if not 0.0 <= self.configuracao.densidade_de_paredes <= 1.0:
            raise ValueError("A densidade de paredes deve estar entre 0 e 1")
        if not 0.0 <= self.configuracao.densidade_de_lixo <= 1.0:
            raise ValueError("A densidade de lixo deve estar entre 0 e 1")

    def _gerar_paredes(self):
        """Sorteia a disposição das paredes internas.

        Returns:
            Matriz booleana [y, x], com a célula inicial sempre livre.
        """
        tamanho = self.configuracao.tamanho_da_grade
        sorteio = self.gerador.random((tamanho, tamanho))
        paredes = sorteio < self.configuracao.densidade_de_paredes

        coluna, linha = self.configuracao.posicao_inicial
        paredes[linha, coluna] = False
        return paredes

    def reset(self):
        """Inicia um novo episódio.

        Redistribui o lixo nas células livres e devolve Marvin à posição inicial.

        Returns:
            Índice do estado percebido na posição inicial.
        """
        self.grade = np.full(self.paredes.shape, Celula.vazio, dtype=np.int8)
        self.grade[self.paredes] = Celula.parede

        sorteio = self.gerador.random(self.paredes.shape)
        tem_lixo = (~self.paredes) & (sorteio < self.configuracao.densidade_de_lixo)
        self.grade[tem_lixo] = Celula.lixo

        self.posicao = tuple(self.configuracao.posicao_inicial)
        self.passo_atual = 0
        return self.obter_estado()

    @property
    def quantidade_de_lixo_restante(self):
        """Quantidade de células com lixo ainda presentes na grade."""
        return int(np.sum(self.grade == Celula.lixo))

    def _conteudo_da_celula(self, coluna, linha):
        """Lê o conteúdo de uma célula, tratando a área fora da grade como parede.

        Args:
            coluna: Coordenada x da célula.
            linha: Coordenada y da célula.

        Returns:
            Valor de Celula da posição consultada.
        """
        tamanho = self.configuracao.tamanho_da_grade
        if not (0 <= coluna < tamanho and 0 <= linha < tamanho):
            return Celula.parede
        return Celula(int(self.grade[linha, coluna]))

    def obter_estado(self):
        """Calcula a percepção de Marvin na posição atual.

        Returns:
            Índice do estado formado pela célula atual e pelas quatro vizinhas.
        """
        coluna, linha = self.posicao
        conteudos = [self._conteudo_da_celula(coluna, linha)]

        for acao in acoes_de_movimento:
            deslocamento_x, deslocamento_y = deslocamento_por_acao[acao]
            conteudos.append(
                self._conteudo_da_celula(coluna + deslocamento_x, linha + deslocamento_y)
            )

        return codificar_percepcao(*conteudos)

    def passo(self, acao):
        """Executa uma ação e avança o ambiente em um passo.

        O fim do episódio por limite de tempo é apenas um truncamento, e não um
        estado terminal. Por isso, o agente deve manter o termo de bootstrapping
        da Eq. (2) mesmo quando terminado for verdadeiro.

        Args:
            acao: Ação a executar, como Acao ou inteiro equivalente.

        Returns:
            Tupla (novo_estado, recompensa, terminado, resultado), em que
            terminado indica que o horizonte H foi atingido e resultado é um
            ResultadoPasso usado para contar eventos.

        Raises:
            RuntimeError: Se o episódio já terminou e reset() não foi chamado.
        """
        if self.passo_atual >= self.configuracao.passos_por_episodio:
            raise RuntimeError("O episódio terminou, chame reset() antes de continuar")

        acao = Acao(acao)

        # A ação aleatória vira um movimento em uma direção sorteada uniformemente
        if acao == Acao.aleatoria:
            indice_da_direcao = self.gerador.integers(len(acoes_de_movimento))
            acao_efetiva = acoes_de_movimento[indice_da_direcao]
        else:
            acao_efetiva = acao

        if acao_efetiva == Acao.coletar:
            recompensa, resultado = self._coletar()
        elif acao_efetiva == Acao.parado:
            recompensa, resultado = self.recompensas.demais_transicoes, ResultadoPasso.outro
        else:
            recompensa, resultado = self._mover(acao_efetiva)

        self.passo_atual += 1
        terminado = self.passo_atual >= self.configuracao.passos_por_episodio
        return self.obter_estado(), recompensa, terminado, resultado

    def _coletar(self):
        """Tenta coletar o lixo da célula atual.

        Returns:
            Tupla (recompensa, resultado) da tentativa de coleta.
        """
        coluna, linha = self.posicao

        if self.grade[linha, coluna] == Celula.lixo:
            self.grade[linha, coluna] = Celula.vazio
            return self.recompensas.coleta_com_sucesso, ResultadoPasso.coleta_com_sucesso

        return self.recompensas.coleta_sem_lixo, ResultadoPasso.coleta_sem_lixo

    def _mover(self, direcao):
        """Tenta mover Marvin uma célula na direção indicada.

        Args:
            direcao: Uma das quatro ações de movimento.

        Returns:
            Tupla (recompensa, resultado) do movimento. Em caso de parede ou
            borda, Marvin permanece na mesma célula.
        """
        coluna, linha = self.posicao
        deslocamento_x, deslocamento_y = deslocamento_por_acao[direcao]
        nova_coluna = coluna + deslocamento_x
        nova_linha = linha + deslocamento_y

        if self._conteudo_da_celula(nova_coluna, nova_linha) == Celula.parede:
            return self.recompensas.movimento_invalido, ResultadoPasso.movimento_invalido

        self.posicao = (nova_coluna, nova_linha)
        return self.recompensas.demais_transicoes, ResultadoPasso.outro

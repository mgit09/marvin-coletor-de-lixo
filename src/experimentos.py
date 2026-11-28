"""Experimentos das Tarefas 3 e 4: execuções repetidas, agregação e tabelas.

Cada configuração (por exemplo, uma agenda de epsilon) é treinada várias vezes
com sementes diferentes, e as métricas ficam empilhadas em um
AgregadoExperimento. As funções de alto nível reproduzem a varredura de epsilon
fixo, a comparação das agendas de decaimento e o teste de sensibilidade de
alpha e gamma usado para congelar os hiperparâmetros.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.agendas_epsilon import (
    criar_agenda_constante,
    criar_agenda_exponencial,
    criar_agenda_harmonica,
)
from src.configuracao import ConfiguracaoAmbiente, ConfiguracaoTreinamento
from src.treinamento import treinar

JANELA_DA_RECOMPENSA_FINAL = 50
VALORES_DE_EPSILON_DA_VARREDURA = (0.0, 0.01, 0.05, 0.1, 0.3, 1.0)
TAXAS_DE_APRENDIZADO_DA_SENSIBILIDADE = (0.05, 0.1, 0.3)
FATORES_DE_DESCONTO_DA_SENSIBILIDADE = (0.8, 0.9, 0.95)

construtores_de_agenda = {
    "constante": criar_agenda_constante,
    "exponencial": criar_agenda_exponencial,
    "harmonica": criar_agenda_harmonica,
}


def formatar_numero(valor, casas_decimais=1):
    """Formata um número com vírgula decimal, como no relatório.

    Args:
        valor: Número a formatar.
        casas_decimais: Quantidade de casas depois da vírgula.

    Returns:
        Texto com o número, por exemplo "391,1".
    """
    return f"{valor:.{casas_decimais}f}".replace(".", ",")


def _formatar_parametro(valor):
    """Formata um hiperparâmetro de forma compacta, com vírgula decimal.

    Args:
        valor: Número a formatar.

    Returns:
        Texto sem zeros desnecessários, por exemplo "0,05".
    """
    return f"{valor:g}".replace(".", ",")


def _desvio_padrao(valores):
    """Calcula o desvio padrão amostral, ou zero quando há uma só execução.

    Args:
        valores: Valores, um por execução.

    Returns:
        Desvio padrão com ddof = 1, ou 0 se houver um único valor.
    """
    valores = np.asarray(valores, dtype=float)
    return float(np.std(valores, ddof=1)) if len(valores) > 1 else 0.0


@dataclass(eq=False)
class AgregadoExperimento:
    """Métricas de uma configuração treinada em várias execuções.

    Todas as matrizes de métricas têm formato (execuções, episódios).

    Attributes:
        nome: Nome legível, usado em legendas e tabelas.
        identificador: Texto sem acentos, usado em nomes de arquivo.
        metadados: Parâmetros da configuração, para validar o uso de cache.
        recompensa_por_episodio: Recompensa total de cada episódio.
        lixos_coletados_por_episodio: Coletas bem-sucedidas.
        coletas_malsucedidas_por_episodio: Coletas em células sem lixo.
        movimentos_invalidos_por_episodio: Colisões com parede ou borda.
        epsilon_por_episodio: Epsilon usado em cada episódio.
        fracao_visitada_por_episodio: Fração de pares (s, a) visitados.
        q_tables: Q-table final de cada execução, formato (execuções, 162, 7).
        contagem_de_visitas: Visitas a cada par (s, a), formato (execuções, 162, 7).
    """

    nome: str
    identificador: str
    metadados: dict
    recompensa_por_episodio: np.ndarray
    lixos_coletados_por_episodio: np.ndarray
    coletas_malsucedidas_por_episodio: np.ndarray
    movimentos_invalidos_por_episodio: np.ndarray
    epsilon_por_episodio: np.ndarray
    fracao_visitada_por_episodio: np.ndarray
    q_tables: np.ndarray
    contagem_de_visitas: np.ndarray

    @classmethod
    def a_partir_dos_resultados(cls, nome, identificador, metadados, resultados):
        """Empilha os resultados de várias execuções em um único agregado.

        Args:
            nome: Nome legível da configuração.
            identificador: Texto sem acentos para nomes de arquivo.
            metadados: Parâmetros da configuração.
            resultados: Lista de ResultadoTreinamento, um por execução.

        Returns:
            AgregadoExperimento com as métricas empilhadas.
        """
        return cls(
            nome=nome,
            identificador=identificador,
            metadados=metadados,
            recompensa_por_episodio=np.stack([r.recompensa_por_episodio for r in resultados]),
            lixos_coletados_por_episodio=np.stack([r.lixos_coletados_por_episodio for r in resultados]),
            coletas_malsucedidas_por_episodio=np.stack(
                [r.coletas_malsucedidas_por_episodio for r in resultados]
            ),
            movimentos_invalidos_por_episodio=np.stack(
                [r.movimentos_invalidos_por_episodio for r in resultados]
            ),
            epsilon_por_episodio=np.stack([r.epsilon_por_episodio for r in resultados]),
            fracao_visitada_por_episodio=np.stack([r.fracao_visitada_por_episodio for r in resultados]),
            q_tables=np.stack([r.agente.q_table for r in resultados]),
            contagem_de_visitas=np.stack([r.agente.contagem_de_visitas for r in resultados]),
        )

    @property
    def numero_de_execucoes(self):
        """Quantidade de execuções agregadas."""
        return self.recompensa_por_episodio.shape[0]

    def recompensa_media_final_por_execucao(self, janela=JANELA_DA_RECOMPENSA_FINAL):
        """Calcula, para cada execução, a recompensa média dos últimos episódios.

        Args:
            janela: Quantidade de episódios finais considerados.

        Returns:
            Vetor com um valor por execução.
        """
        return self.recompensa_por_episodio[:, -janela:].mean(axis=1)

    @property
    def melhor_recompensa_por_execucao(self):
        """Maior recompensa de um único episódio em cada execução."""
        return self.recompensa_por_episodio.max(axis=1)

    @property
    def fracao_visitada_final_por_execucao(self):
        """Fração de pares (s, a) visitados ao final de cada execução."""
        return self.fracao_visitada_por_episodio[:, -1]

    def salvar(self, caminho):
        """Salva o agregado em um arquivo .npz comprimido.

        Args:
            caminho: Caminho do arquivo, terminado em .npz.
        """
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            caminho,
            nome=self.nome,
            identificador=self.identificador,
            metadados=json.dumps(self.metadados),
            recompensa_por_episodio=self.recompensa_por_episodio,
            lixos_coletados_por_episodio=self.lixos_coletados_por_episodio,
            coletas_malsucedidas_por_episodio=self.coletas_malsucedidas_por_episodio,
            movimentos_invalidos_por_episodio=self.movimentos_invalidos_por_episodio,
            epsilon_por_episodio=self.epsilon_por_episodio,
            fracao_visitada_por_episodio=self.fracao_visitada_por_episodio,
            q_tables=self.q_tables,
            contagem_de_visitas=self.contagem_de_visitas,
        )

    @classmethod
    def carregar(cls, caminho):
        """Carrega um agregado salvo por salvar().

        Args:
            caminho: Caminho do arquivo .npz.

        Returns:
            AgregadoExperimento com os dados do arquivo.
        """
        with np.load(caminho, allow_pickle=False) as dados:
            return cls(
                nome=str(dados["nome"]),
                identificador=str(dados["identificador"]),
                metadados=json.loads(str(dados["metadados"])),
                recompensa_por_episodio=dados["recompensa_por_episodio"],
                lixos_coletados_por_episodio=dados["lixos_coletados_por_episodio"],
                coletas_malsucedidas_por_episodio=dados["coletas_malsucedidas_por_episodio"],
                movimentos_invalidos_por_episodio=dados["movimentos_invalidos_por_episodio"],
                epsilon_por_episodio=dados["epsilon_por_episodio"],
                fracao_visitada_por_episodio=dados["fracao_visitada_por_episodio"],
                q_tables=dados["q_tables"],
                contagem_de_visitas=dados["contagem_de_visitas"],
            )


def _treinar_uma_execucao(tarefa):
    """Treina uma execução, em formato adequado para processos paralelos.

    Args:
        tarefa: Tupla (nome_da_agenda, parametros_da_agenda,
            configuracao_treinamento, configuracao_ambiente, semente).

    Returns:
        ResultadoTreinamento da execução.
    """
    nome_da_agenda, parametros_da_agenda, configuracao_treinamento, configuracao_ambiente, semente = tarefa
    agenda = construtores_de_agenda[nome_da_agenda](**parametros_da_agenda)
    return treinar(agenda, configuracao_treinamento, configuracao_ambiente, semente=semente)


def _montar_metadados(
    nome_da_agenda, parametros_da_agenda, numero_de_execucoes, configuracao_treinamento, configuracao_ambiente
):
    """Reúne em um dicionário os parâmetros que definem uma configuração.

    Args:
        nome_da_agenda: Chave de construtores_de_agenda.
        parametros_da_agenda: Argumentos do construtor da agenda.
        numero_de_execucoes: Quantidade de execuções.
        configuracao_treinamento: Hiperparâmetros do treinamento.
        configuracao_ambiente: Parâmetros do ambiente.

    Returns:
        Dicionário compatível com JSON.
    """
    metadados = {
        "nome_da_agenda": nome_da_agenda,
        "parametros_da_agenda": parametros_da_agenda,
        "numero_de_execucoes": numero_de_execucoes,
        "configuracao_treinamento": asdict(configuracao_treinamento),
        "configuracao_ambiente": asdict(configuracao_ambiente),
    }
    # Passa por JSON para que tuplas virem listas e a comparação com o cache seja exata
    return json.loads(json.dumps(metadados))


def executar_configuracao(
    nome,
    identificador,
    nome_da_agenda,
    parametros_da_agenda,
    numero_de_execucoes=10,
    configuracao_treinamento=None,
    configuracao_ambiente=None,
    numero_de_processos=None,
    pasta_de_dados=None,
):
    """Treina uma configuração em várias execuções com sementes consecutivas.

    As sementes vão de semente_base até semente_base + execuções - 1. Como são
    as mesmas para todas as configurações, a execução i de cada configuração
    enfrenta o mesmo mapa, o que torna as comparações pareadas.

    Args:
        nome: Nome legível da configuração.
        identificador: Texto sem acentos para o nome do arquivo de cache.
        nome_da_agenda: Chave de construtores_de_agenda.
        parametros_da_agenda: Argumentos do construtor da agenda.
        numero_de_execucoes: Quantidade de execuções com sementes diferentes.
        configuracao_treinamento: Hiperparâmetros. Usa o padrão se for None.
        configuracao_ambiente: Parâmetros do ambiente. Usa o padrão se for None.
        numero_de_processos: Processos em paralelo. Usa o número de núcleos se for None.
        pasta_de_dados: Pasta de cache. Se o arquivo da configuração existir e tiver
            os mesmos parâmetros, ele é carregado em vez de treinar de novo.

    Returns:
        AgregadoExperimento com as métricas das execuções.
    """
    configuracao_treinamento = configuracao_treinamento or ConfiguracaoTreinamento()
    configuracao_ambiente = configuracao_ambiente or ConfiguracaoAmbiente()
    metadados = _montar_metadados(
        nome_da_agenda,
        parametros_da_agenda,
        numero_de_execucoes,
        configuracao_treinamento,
        configuracao_ambiente,
    )

    caminho_do_cache = Path(pasta_de_dados) / f"{identificador}.npz" if pasta_de_dados else None
    if caminho_do_cache is not None and caminho_do_cache.exists():
        agregado = AgregadoExperimento.carregar(caminho_do_cache)
        if agregado.metadados == metadados:
            agregado.nome = nome
            return agregado

    sementes = [configuracao_treinamento.semente_base + indice for indice in range(numero_de_execucoes)]
    tarefas = [
        (nome_da_agenda, parametros_da_agenda, configuracao_treinamento, configuracao_ambiente, semente)
        for semente in sementes
    ]

    if numero_de_processos is None:
        numero_de_processos = min(os.cpu_count() or 1, numero_de_execucoes)

    if numero_de_processos > 1:
        with ProcessPoolExecutor(max_workers=numero_de_processos) as executor:
            resultados = list(executor.map(_treinar_uma_execucao, tarefas))
    else:
        resultados = [_treinar_uma_execucao(tarefa) for tarefa in tarefas]

    agregado = AgregadoExperimento.a_partir_dos_resultados(nome, identificador, metadados, resultados)
    if caminho_do_cache is not None:
        agregado.salvar(caminho_do_cache)
    return agregado


def varredura_de_epsilon(valores_de_epsilon=VALORES_DE_EPSILON_DA_VARREDURA, **argumentos):
    """Treina Marvin com cada valor de epsilon fixo (Tarefa 4a).

    Args:
        valores_de_epsilon: Valores de epsilon a comparar.
        **argumentos: Repassados a executar_configuracao (execuções, configurações,
            processos e pasta de dados).

    Returns:
        Dicionário nome -> AgregadoExperimento, na ordem dos valores.
    """
    agregados = {}
    for valor in valores_de_epsilon:
        texto = _formatar_parametro(valor)
        nome = f"ε = {texto}"
        identificador = f"varredura_epsilon_{texto.replace(',', '_')}"
        agregados[nome] = executar_configuracao(
            nome, identificador, "constante", {"epsilon": valor}, **argumentos
        )
    return agregados


def comparar_agendas(**argumentos):
    """Treina Marvin com as três agendas de decaimento da Tarefa 4b.

    Args:
        **argumentos: Repassados a executar_configuracao.

    Returns:
        Dicionário nome -> AgregadoExperimento, na ordem constante, exponencial e harmônica.
    """
    especificacoes = (
        ("Constante (ε = 0,1)", "agenda_constante", "constante", {"epsilon": 0.1}),
        (
            "Exponencial (d = 0,99)",
            "agenda_exponencial",
            "exponencial",
            {"epsilon_inicial": 1.0, "decaimento": 0.99, "epsilon_minimo": 0.01},
        ),
        ("Harmônica (1/k)", "agenda_harmonica", "harmonica", {"epsilon_minimo": 0.01}),
    )
    return {
        nome: executar_configuracao(nome, identificador, nome_da_agenda, parametros, **argumentos)
        for nome, identificador, nome_da_agenda, parametros in especificacoes
    }


def analisar_sensibilidade(
    taxas_de_aprendizado=TAXAS_DE_APRENDIZADO_DA_SENSIBILIDADE,
    fatores_de_desconto=FATORES_DE_DESCONTO_DA_SENSIBILIDADE,
    configuracao_treinamento=None,
    **argumentos,
):
    """Varia alpha e gamma com a agenda exponencial, para escolher os hiperparâmetros.

    Args:
        taxas_de_aprendizado: Valores de alpha a testar.
        fatores_de_desconto: Valores de gamma a testar.
        configuracao_treinamento: Configuração base. Usa o padrão se for None.
        **argumentos: Repassados a executar_configuracao.

    Returns:
        Dicionário nome -> AgregadoExperimento, com uma entrada por par (alpha, gamma).
    """
    configuracao_base = configuracao_treinamento or ConfiguracaoTreinamento()
    agregados = {}
    for taxa in taxas_de_aprendizado:
        for fator in fatores_de_desconto:
            configuracao = replace(configuracao_base, taxa_de_aprendizado=taxa, fator_de_desconto=fator)
            nome = f"α = {_formatar_parametro(taxa)}, γ = {_formatar_parametro(fator)}"
            identificador = "sensibilidade_alfa_{}_gama_{}".format(
                _formatar_parametro(taxa).replace(",", "_"), _formatar_parametro(fator).replace(",", "_")
            )
            agregados[nome] = executar_configuracao(
                nome,
                identificador,
                "exponencial",
                {"epsilon_inicial": 1.0, "decaimento": 0.99, "epsilon_minimo": 0.01},
                configuracao_treinamento=configuracao,
                **argumentos,
            )
    return agregados


def montar_tabela_resumo(agregados, janela=JANELA_DA_RECOMPENSA_FINAL):
    """Resume cada configuração com média e desvio padrão entre as execuções.

    As colunas correspondem à tabela da Tarefa 4a: recompensa média dos últimos
    episódios, melhor episódio e porcentagem da Q-table visitada.

    Args:
        agregados: Dicionário nome -> AgregadoExperimento.
        janela: Quantidade de episódios finais da recompensa média.

    Returns:
        DataFrame indexado pelo nome da configuração.
    """
    linhas = []
    for nome, agregado in agregados.items():
        medias_finais = agregado.recompensa_media_final_por_execucao(janela)
        melhores = agregado.melhor_recompensa_por_execucao
        porcentagens = 100.0 * agregado.fracao_visitada_final_por_execucao
        linhas.append(
            {
                "configuracao": nome,
                "recompensa_media_final": float(medias_finais.mean()),
                "recompensa_media_final_desvio": _desvio_padrao(medias_finais),
                "melhor_episodio": float(melhores.mean()),
                "melhor_episodio_desvio": _desvio_padrao(melhores),
                "porcentagem_visitada": float(porcentagens.mean()),
                "porcentagem_visitada_desvio": _desvio_padrao(porcentagens),
                "numero_de_execucoes": agregado.numero_de_execucoes,
            }
        )
    return pd.DataFrame(linhas).set_index("configuracao")


def montar_tabela_de_sensibilidade(agregados, janela=JANELA_DA_RECOMPENSA_FINAL):
    """Organiza o teste de sensibilidade com uma linha por par (alpha, gamma).

    Args:
        agregados: Dicionário nome -> AgregadoExperimento, de analisar_sensibilidade.
        janela: Quantidade de episódios finais da recompensa média.

    Returns:
        DataFrame com alpha, gamma, recompensa média final, desvio e número de execuções.
    """
    linhas = []
    for agregado in agregados.values():
        configuracao = agregado.metadados["configuracao_treinamento"]
        medias_finais = agregado.recompensa_media_final_por_execucao(janela)
        linhas.append(
            {
                "taxa_de_aprendizado": configuracao["taxa_de_aprendizado"],
                "fator_de_desconto": configuracao["fator_de_desconto"],
                "recompensa_media_final": float(medias_finais.mean()),
                "recompensa_media_final_desvio": _desvio_padrao(medias_finais),
                "numero_de_execucoes": agregado.numero_de_execucoes,
            }
        )
    return pd.DataFrame(linhas)


def _nome_em_latex(nome):
    """Converte o nome de uma configuração em texto para LaTeX.

    Args:
        nome: Nome legível, por exemplo "ε = 0,05" ou "Harmônica (1/k)".

    Returns:
        Texto com o epsilon em modo matemático quando aplicável.
    """
    if nome.startswith("ε = "):
        valor = nome[len("ε = "):].replace(",", "{,}")
        return f"$\\epsilon = {valor}$"
    return nome


def _celula_em_latex(media, desvio, casas_decimais=1):
    """Formata uma célula "média ± desvio" em LaTeX, com vírgula decimal.

    Args:
        media: Valor médio.
        desvio: Desvio padrão.
        casas_decimais: Casas decimais de ambos.

    Returns:
        Texto em modo matemático, por exemplo "$391{,}1 \\pm 12{,}3$".
    """
    media_formatada = formatar_numero(media, casas_decimais).replace(",", "{,}")
    desvio_formatado = formatar_numero(desvio, casas_decimais).replace(",", "{,}")
    return f"${media_formatada} \\pm {desvio_formatado}$"


def tabela_em_latex(tabela, titulo_da_primeira_coluna="Configuração"):
    """Gera o ambiente tabular em LaTeX (booktabs) da tabela resumo.

    O ambiente table, a legenda e o rótulo ficam a cargo do relatório, que deve
    informar o número de execuções agregadas.

    Args:
        tabela: DataFrame de montar_tabela_resumo.
        titulo_da_primeira_coluna: Cabeçalho da coluna com os nomes.

    Returns:
        Texto LaTeX com o ambiente tabular.
    """
    linhas = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        f"{titulo_da_primeira_coluna} & Recomp. média (últimos 50 ep.) & Melhor episódio "
        "& \\% da Q-table visitada \\\\",
        "\\midrule",
    ]
    for nome, linha in tabela.iterrows():
        celulas = [
            _nome_em_latex(nome),
            _celula_em_latex(linha["recompensa_media_final"], linha["recompensa_media_final_desvio"]),
            _celula_em_latex(linha["melhor_episodio"], linha["melhor_episodio_desvio"]),
            _celula_em_latex(linha["porcentagem_visitada"], linha["porcentagem_visitada_desvio"]),
        ]
        linhas.append(" & ".join(celulas) + " \\\\")
    linhas += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(linhas)

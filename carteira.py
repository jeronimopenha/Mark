import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_inicio = "2016-01-01"

# Defina os pesos fixos para testar manualmente
pesos_teste = {
    'IFIX': 0.0487,
    'IBOV': 0.2025,
    'IVVB11': 0.2164,
    'BTC': 0.0586,
    'SELIC': 0.4738,
}

# Ativos
tickers = {
    'IFIX': 'XFIX11.SA',
    'IBOV': 'BOVA11.SA',
    'IVVB11': 'IVVB11.SA',
    'BTC': 'HASH11.SA',
}

# Taxa Selic anual e mensal (constante)
selic_anual = 0.15
selic_mensal = (1 + selic_anual) ** (1 / 12) - 1

# Baixar dados
dados = yf.download(list(tickers.values()), start=data_inicio, interval="1mo", auto_adjust=True, threads=False)[
    'Close']
dados.dropna(inplace=True)

# Retornos mensais
retornos = dados.pct_change().dropna()
retornos['SELIC'] = selic_mensal  # constante mensal
tickers['SELIC'] = 'SELIC'

# Estatísticas
media_retornos = retornos.mean()
cov_retornos = retornos.cov()

# Garante que a soma é 1.0 (opcional, apenas alerta)
soma = sum(pesos_teste.values())
if not np.isclose(soma, 1.0):
    raise ValueError(f"A soma dos pesos = {soma:.4f}, deve ser 1.0")

# Cria vetor de pesos na ordem dos tickers
vetor_pesos = np.array([pesos_teste[t] for t in tickers.keys()])

# Calcula retorno, risco e Sharpe da alocação
retorno_teste = np.dot(vetor_pesos, media_retornos) * 12
risco_teste = np.sqrt(np.dot(vetor_pesos.T, np.dot(cov_retornos * 12, vetor_pesos)))
sharpe_teste = (retorno_teste - selic_anual) / risco_teste

# Impressão
print("\n📌 Análise da Carteira com Alocação Manual:")
print("Retorno esperado anual: {:.2f}%".format(retorno_teste * 100))
print("Risco anual (volatilidade): {:.2f}%".format(risco_teste * 100))
print("Sharpe Ratio: {:.2f}".format(sharpe_teste))
print("Pesos:")
for ativo, peso in pesos_teste.items():
    print(f"  {ativo}: {peso*100:.2f}%")
print("Intervalos de confiança:")
print("  68%: {:.2f}% a {:.2f}%".format((retorno_teste - risco_teste)*100, (retorno_teste + risco_teste)*100))
print("  95%: {:.2f}% a {:.2f}%".format((retorno_teste - 2*risco_teste)*100, (retorno_teste + 2*risco_teste)*100))
print("  99.7%: {:.2f}% a {:.2f}%".format((retorno_teste - 3*risco_teste)*100, (retorno_teste + 3*risco_teste)*100))

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_inicio = "2016-01-01"

pesos_fixos = {
    'IFIX': 0.0,
    'IBOV': 0.0,
    'IVVB11': 0.0,
    'BTC': 0.0,
    'SELIC': 0.0
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

# Simular 10.000 portfólios
n_simul = 10000
resultados = np.zeros((n_simul, 3))
pesos_lista = []

for i in range(n_simul):
    ativos_fixos = [k for k, v in pesos_fixos.items() if v > 0]
    ativos_variaveis = [k for k, v in pesos_fixos.items() if v == 0]
    soma_fixos = sum(pesos_fixos.values())
    sobra = 1.0 - soma_fixos

    # Distribui o restante entre os ativos variáveis
    pesos_aleatorios = np.random.random(len(ativos_variaveis))
    pesos_aleatorios /= pesos_aleatorios.sum()
    pesos_aleatorios *= sobra

    # Montar vetor de pesos final na ordem dos tickers
    pesos = []
    for ativo in tickers:
        if pesos_fixos[ativo] > 0:
            pesos.append(pesos_fixos[ativo])
        else:
            pesos.append(pesos_aleatorios[0])
            pesos_aleatorios = pesos_aleatorios[1:]
    pesos = np.array(pesos)
    pesos_lista.append(pesos)

    retorno = np.dot(pesos, media_retornos) * 12
    risco = np.sqrt(np.dot(pesos.T, np.dot(cov_retornos * 12, pesos)))
    risco = max(risco, 1e-6)
    # sharpe = (retorno - selic_anual) / risco
    sharpe = (retorno - selic_anual) / risco

    resultados[i] = [retorno, risco, sharpe]


# DataFrame e melhor portfólio
df_resultados = pd.DataFrame(resultados, columns=['Retorno', 'Risco', 'Sharpe'])
melhor = df_resultados.loc[df_resultados['Sharpe'].idxmax()]
mais_seguro = df_resultados.loc[df_resultados['Risco'].idxmin()]

# Adicionar os pesos simulados ao DataFrame
df_pesos = pd.DataFrame(pesos_lista, columns=list(tickers.keys()))
df_completo = pd.concat([df_resultados, df_pesos], axis=1)
print(df_completo)
# Ordenar por risco e retorno
df_ordenado = df_completo.sort_values(by=['Risco', 'Retorno'])



# Selecionar envelope superior
envelope = []
max_retorno = -np.inf
for _, row in df_ordenado.iterrows():
    if row['Retorno'] > max_retorno:
        envelope.append(row)
        max_retorno = row['Retorno']

df_envelope = pd.DataFrame(envelope)

### 🔽🔽🔽 FILTROS OPCIONAIS 🔽🔽🔽

# df_envelope = df_envelope[df_envelope['Sharpe'] > 0.5]
#df_envelope = df_envelope[df_envelope['Sharpe'] > -1]  # ou até mesmo remover o filtro

# Apenas portfólios onde retorno compensa o risco 1:1
df_envelope = df_envelope[df_envelope['Retorno'] - df_envelope['Risco'] >= 0]

# Apenas portfólios onde retorno ≥ 50% do risco
# df_envelope = df_envelope[df_envelope['Retorno'] - df_envelope['Risco'] * 0.5 >= 0]

# Apenas portfólios com Sharpe maior que 0.5
#df_envelope = df_envelope[df_envelope['Sharpe'] > 0.5]

# Apenas portfólios com risco menor que 20%
# df_envelope = df_envelope[df_envelope['Risco'] <= 0.20]

# Adicionar colunas com variações esperadas (intervalos de confiança)
for df in [df_completo, df_envelope]:
    if not df.empty:
        df['min_68'] = df['Retorno'] - df['Risco']
        df['max_68'] = df['Retorno'] + df['Risco']

        df['min_95'] = df['Retorno'] - 2 * df['Risco']
        df['max_95'] = df['Retorno'] + 2 * df['Risco']

        df['min_997'] = df['Retorno'] - 3 * df['Risco']
        df['max_997'] = df['Retorno'] + 3 * df['Risco']

# Salvar todos os portfólios simulados
df_completo.to_csv("todos_os_portfolios.csv", index=False)

# Verificar e salvar se houver resultado
if df_envelope.empty:
    print("⚠️ Nenhum portfólio atendeu ao filtro escolhido.")
else:
    df_envelope.to_csv("fronteira_superior.csv", index=False)
    print(f"✅ {len(df_envelope)} portfólios salvos em 'fronteira_superior.csv' com o filtro aplicado.")

# Encontrar ponto T (maior Sharpe) e ponto mais seguro (menor risco)
idx_t = df_resultados['Sharpe'].idxmax()
idx_seguro = df_resultados['Risco'].idxmin()

# Obter pesos correspondentes
pesos_t = df_completo.loc[idx_t, tickers.keys()]
pesos_seguro = df_completo.loc[idx_seguro, tickers.keys()]

# Ponto da Selic no gráfico
retorno_selic = selic_anual
risco_selic = 0

# Ponto T (portfólio ótimo)
retorno_t = melhor['Retorno']
risco_t = melhor['Risco']

# Gráfico
plt.figure(figsize=(10, 6))
plt.scatter(
    df_resultados['Risco'],
    df_resultados['Retorno'],
    c=df_resultados['Sharpe'],
    cmap='viridis', s=10
)

plt.colorbar(label='Sharpe Ratio')
plt.scatter(
    melhor['Risco'],
    melhor['Retorno'],
    c='red', s=60,
    label='Portfólio T'
)
plt.scatter(
    mais_seguro['Risco'],
    mais_seguro['Retorno'],
    c='blue',
    s=60,
    label='Menor Risco'
)
# Traçar a CML
plt.plot(
    [risco_selic, risco_t],
    [retorno_selic, retorno_t],
    color='orange',
    linestyle='--',
    label='CML (linha de mercado de capital)'
)
if not df_envelope.empty:
    plt.scatter(
        df_envelope['Risco'],
        df_envelope['Retorno'],
        c='black',  # ou outra cor que contraste
        s=20,
        label='Fronteira Filtrada'
    )
plt.xlabel('Risco (Volatilidade Anual)')
plt.ylabel('Retorno Esperado Anual')
plt.title('Fronteira Eficiente (Markowitz)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Printar ponto T (maior Sharpe)
retorno_t = df_resultados.loc[idx_t, 'Retorno']
risco_t = df_resultados.loc[idx_t, 'Risco']
print("\n🔺 Portfólio T (Maior Sharpe):")
print("Retorno esperado anual: {:.2f}%".format(retorno_t * 100))
print("Risco anual: {:.2f}%".format(risco_t * 100))
print("Sharpe Ratio: {:.2f}".format(df_resultados.loc[idx_t, 'Sharpe']))
print("Pesos:")
print(pesos_t.apply(lambda x: f"{x * 100:.2f}%"))
print("Intervalos de confiança:")
print("  68%: {:.2f}% a {:.2f}%".format((retorno_t - risco_t) * 100, (retorno_t + risco_t) * 100))
print("  95%: {:.2f}% a {:.2f}%".format((retorno_t - 2 * risco_t) * 100, (retorno_t + 2 * risco_t) * 100))
print("  99.7%: {:.2f}% a {:.2f}%".format((retorno_t - 3 * risco_t) * 100, (retorno_t + 3 * risco_t) * 100))

# Printar portfólio mais seguro
retorno_s = df_resultados.loc[idx_seguro, 'Retorno']
risco_s = df_resultados.loc[idx_seguro, 'Risco']
print("\n🔵 Portfólio Mais Seguro (Menor Risco):")
print("Retorno esperado anual: {:.2f}%".format(retorno_s * 100))
print("Risco anual: {:.2f}%".format(risco_s * 100))
print("Sharpe Ratio: {:.2f}".format(df_resultados.loc[idx_seguro, 'Sharpe']))
print("Pesos:")
print(pesos_seguro.apply(lambda x: f"{x * 100:.2f}%"))
print("Intervalos de confiança:")
print("  68%: {:.2f}% a {:.2f}%".format((retorno_s - risco_s) * 100, (retorno_s + risco_s) * 100))
print("  95%: {:.2f}% a {:.2f}%".format((retorno_s - 2 * risco_s) * 100, (retorno_s + 2 * risco_s) * 100))
print("  99.7%: {:.2f}% a {:.2f}%".format((retorno_s - 3 * risco_s) * 100, (retorno_s + 3 * risco_s) * 100))

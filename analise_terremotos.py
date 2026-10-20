# Trabalho Prático: Variáveis Aleatórias e Inferência Estatística
# Tema: número de terremotos de magnitude >= 5 por dia no mundo (USGS)
# Requisitos: pip install pandas scipy matplotlib
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ---------------- 1) DADOS ----------------
# Baixa do USGS os terremotos de magnitude >= 5 (ou lê CSVs: python analise_terremotos.py 2024.csv 2025.csv)
def contagem_por_dia(ano, arquivo=None):
    url = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv"
           f"&starttime={ano}-01-01&endtime={ano + 1}-01-01&minmagnitude=5")
    df = pd.read_csv(arquivo or url)
    df = df[df["type"] == "earthquake"]                      # só terremotos
    dia = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None).dt.normalize()
    dias = pd.date_range(f"{ano}-01-01", f"{ano}-12-31")
    return dia.value_counts().reindex(dias, fill_value=0)    # dias sem terremoto = 0

arq = sys.argv[1:3] + [None, None]
x24 = contagem_por_dia(2024, arq[0])
x25 = contagem_por_dia(2025, arq[1])

# ---------------- 2) ANÁLISE DESCRITIVA ----------------
tabela1 = pd.DataFrame({
    "2024": [len(x24), x24.mean(), x24.median(), x24.mode()[0], x24.var(ddof=0), x24.std(ddof=0), x24.min(), x24.max()],
    "2025": [len(x25), x25.mean(), x25.median(), x25.mode()[0], x25.var(ddof=0), x25.std(ddof=0), x25.min(), x25.max()],
}, index=["n (dias)", "média", "mediana", "moda", "variância", "desvio padrão", "mínimo", "máximo"])
print("TABELA 1 - Medidas descritivas\n", tabela1.round(2), "\n")

# ---------------- 3) DISTRIBUIÇÃO: POISSON (2025) ----------------
n, lam = len(x25), x25.mean()
ks = np.arange(0, 16)                                        # 0..14 e "15 ou mais"
obs = np.array([(x25 == k).sum() for k in ks]); obs[-1] = (x25 >= 15).sum()
prob = stats.poisson.pmf(ks, lam); prob[-1] = stats.poisson.sf(14, lam)
esp = n * prob
tabela2 = pd.DataFrame({"dias observados": obs, "dias esperados (Poisson)": esp.round(1)}, index=ks)
print(f"TABELA 2 - Observado x Poisson (lambda = {lam:.2f})\n", tabela2, "\n")
print(f"variância / média = {x25.var(ddof=1) / lam:.1f}  (na Poisson seria perto de 1)")

# teste qui-quadrado de aderência (agrupa classes até esperado >= 5; gl = classes - 1 - 1)
go, ge, ao, ae = [], [], 0, 0
for o, e in zip(obs, esp):
    ao += o; ae += e
    if ae >= 5: go.append(ao); ge.append(ae); ao, ae = 0, 0
go[-1] += ao; ge[-1] += ae
qui2 = ((np.array(go) - np.array(ge)) ** 2 / np.array(ge)).sum()
gl = len(go) - 2
print(f"qui-quadrado = {qui2:.1f} | gl = {gl} | p-valor = {stats.chi2.sf(qui2, gl):.2e}\n")

# ---------------- 4) INTERVALO DE CONFIANÇA (95%) ----------------
for nome, x in (("2024", x24), ("2025", x25)):
    lo, hi = stats.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
    print(f"IC 95% para a média de {nome}: [{lo:.2f} ; {hi:.2f}]")

# ---------------- 5) TESTE DE HIPÓTESE (Welch): 2025 x 2024 ----------------
t, p = stats.ttest_ind(x25, x24, equal_var=False)
print(f"\nH0: média 2025 = média 2024 | H1: diferentes | alfa = 0,05")
print(f"t = {t:.2f} | p-valor = {p:.4f} ->", "rejeita H0" if p < 0.05 else "não rejeita H0")

# ---------------- 6) GRÁFICOS ----------------
plt.figure(figsize=(9, 4))
plt.plot(x25.index, x25.values, lw=0.8); plt.axhline(lam, color="red", ls="--", label=f"média = {lam:.2f}")
plt.title("Terremotos M>=5 por dia em 2025"); plt.ylabel("terremotos no dia"); plt.legend()
plt.tight_layout(); plt.savefig("grafico_serie_diaria.png", dpi=150); plt.close()

plt.figure(figsize=(8, 4.5)); w = 0.4
plt.bar(ks - w / 2, obs, w, label="Observado"); plt.bar(ks + w / 2, esp, w, label="Esperado (Poisson)")
plt.xticks(ks, [str(k) for k in ks[:-1]] + ["15+"])
plt.xlabel("terremotos no dia"); plt.ylabel("dias"); plt.title("Observado x Poisson (2025)"); plt.legend()
plt.tight_layout(); plt.savefig("grafico_observado_vs_poisson.png", dpi=150); plt.close()

plt.figure(figsize=(5, 4.5)); plt.boxplot([x24.values, x25.values], tick_labels=["2024", "2025"])
plt.title("Terremotos por dia"); plt.tight_layout(); plt.savefig("grafico_boxplot.png", dpi=150); plt.close()

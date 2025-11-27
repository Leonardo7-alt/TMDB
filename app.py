import os
import ast
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================
# CONFIGURAÇÕES DO DASHBOARD
# =========================
st.set_page_config(
    page_title="Dashboard TMDB – Análise de Filmes e Séries",
    layout="wide",
    page_icon="🎬"
)

st.title("🎬 Dashboard TMDB – Filmes e Séries")
st.markdown(
    """
    Painel interativo com dados coletados da **API pública do TMDB**.<br>
    Use os filtros laterais para explorar padrões por ano, tipo, país, gêneros, popularidade e nota.
    """,
    unsafe_allow_html=True,
)

# =========================
# CARREGAMENTO DOS DADOS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DF_MODELAGEM_PATH = os.path.join(DATA_DIR, "df_modelagem.csv")


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Garantir tipos básicos
    if "ano_lancamento" in df.columns:
        df["ano_lancamento"] = pd.to_numeric(df["ano_lancamento"], errors="coerce")

    if "genre_ids" in df.columns:
        def parse_genre_ids(x):
            if isinstance(x, list):
                return x
            if isinstance(x, str):
                try:
                    v = ast.literal_eval(x)
                    return v if isinstance(v, list) else None
                except Exception:
                    return None
            return None

        df["genre_ids"] = df["genre_ids"].apply(parse_genre_ids)

    # Se generos_nomes vier como string de lista, converter
    if "generos_nomes" in df.columns:
        def parse_genre_names(x):
            if isinstance(x, list):
                return x
            if isinstance(x, str):
                try:
                    v = ast.literal_eval(x)
                    return v if isinstance(v, list) else []
                except Exception:
                    return []
            return []

        df["generos_nomes"] = df["generos_nomes"].apply(parse_genre_names)

    # Colunas auxiliares
    if "generos_nomes" in df.columns:
        df["generos_str"] = df["generos_nomes"].apply(lambda lst: ", ".join(lst) if lst else "")

    return df


if not os.path.exists(DF_MODELAGEM_PATH):
    st.error("❌ Arquivo `df_modelagem.csv` não encontrado. Execute primeiro o notebook de coleta e tratamento.")
    st.stop()

df = load_data(DF_MODELAGEM_PATH)


# =========================
# SIDEBAR – FILTROS
# =========================
st.sidebar.header("🔎 Filtros")

# Ano
if "ano_lancamento" in df.columns:
    anos_validos = sorted(df["ano_lancamento"].dropna().unique().tolist())
    ano_min, ano_max = int(min(anos_validos)), int(max(anos_validos))
    ano_range = st.sidebar.slider(
        "Ano de lançamento",
        min_value=ano_min,
        max_value=ano_max,
        value=(ano_min, ano_max),
        step=1,
    )
else:
    ano_range = (None, None)

# Tipo
tipos = df["tipo"].dropna().unique().tolist() if "tipo" in df.columns else []
tipo_sel = st.sidebar.multiselect("Tipo", options=tipos, default=tipos)

# Nota
nota_min = float(df["vote_average"].min()) if "vote_average" in df.columns else 0.0
nota_max = float(df["vote_average"].max()) if "vote_average" in df.columns else 10.0
nota_range = st.sidebar.slider(
    "Nota média",
    min_value=float(round(nota_min, 1)),
    max_value=float(round(nota_max, 1)),
    value=(float(round(nota_min, 1)), float(round(nota_max, 1))),
)

# Popularidade (opcional)
if "popularity" in df.columns:
    pop_min = float(df["popularity"].min())
    pop_max = float(df["popularity"].max())
    pop_range = st.sidebar.slider(
        "Popularidade",
        min_value=float(round(pop_min, 1)),
        max_value=float(round(pop_max, 1)),
        value=(float(round(pop_min, 1)), float(round(pop_max, 1))),
    )
else:
    pop_range = (None, None)

# =========================
# APLICAR FILTROS
# =========================
df_filt = df.copy()

if ano_range != (None, None) and "ano_lancamento" in df_filt.columns:
    df_filt = df_filt[df_filt["ano_lancamento"].between(ano_range[0], ano_range[1])]

if tipo_sel:
    df_filt = df_filt[df_filt["tipo"].isin(tipo_sel)]

df_filt = df_filt[
    df_filt["vote_average"].between(nota_range[0], nota_range[1])
]

if pop_range != (None, None) and "popularity" in df_filt.columns:
    df_filt = df_filt[
        df_filt["popularity"].between(pop_range[0], pop_range[1])
    ]

if df_filt.empty:
    st.warning("Nenhum título encontrado com os filtros selecionados.")
    st.stop()


# =========================
# KPIs – MÉTRICAS PRINCIPAIS
# =========================
st.markdown("## 📌 Indicadores Gerais (dados filtrados)")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Total de títulos", len(df_filt))

with c2:
    st.metric("Nota média", f"{df_filt['vote_average'].mean():.2f}")

with c3:
    if "popularity" in df_filt.columns:
        st.metric("Popularidade média", f"{df_filt['popularity'].mean():.2f}")
    else:
        st.metric("Popularidade média", "N/D")

with c4:
    if "tamanho_sinopse" in df_filt.columns:
        st.metric("Tam. médio da sinopse", f"{df_filt['tamanho_sinopse'].mean():.1f} palavras")
    else:
        st.metric("Tam. médio da sinopse", "N/D")

st.markdown("---")


# =========================
# LAYOUT EM ABAS
# =========================
aba_geral, aba_mapa, aba_generos, aba_top, aba_corr, aba_tabela = st.tabs(
    ["📊 Visão Geral", "🌍 Mapa Global", "🎭 Gêneros", "🏆 Top Títulos", "📈 Correlação", "📄 Tabela"]
)


# ---------------------------------
# ABA: VISÃO GERAL
# ---------------------------------
with aba_geral:
    col1, col2 = st.columns(2)

    with col1:
        if "ano_lancamento" in df_filt.columns:
            fig_ano = px.histogram(
                df_filt,
                x="ano_lancamento",
                nbins=25,
                title="📅 Distribuição por Ano de Lançamento",
                color_discrete_sequence=["#636EFA"],
            )
            fig_ano.update_layout(template="plotly_white")
            st.plotly_chart(fig_ano, use_container_width=True)

    with col2:
        fig_nota = px.histogram(
            df_filt,
            x="vote_average",
            nbins=20,
            title="⭐ Distribuição de Notas Médias",
            color_discrete_sequence=["#EF553B"],
        )
        fig_nota.update_layout(template="plotly_white")
        st.plotly_chart(fig_nota, use_container_width=True)

    st.markdown("### 🔥 Popularidade x Nota Média")

    if "popularidade_log" in df_filt.columns and "popularity" in df_filt.columns:
        fig_scatter = px.scatter(
            df_filt,
            x="popularidade_log",
            y="vote_average",
            size="popularity",
            hover_name="titulo_unico",
            opacity=0.7,
            title="Popularidade (log) x Nota Média",
            color="tipo",
        )
        fig_scatter.update_traces(marker=dict(line=dict(width=0)))
        fig_scatter.update_layout(template="plotly_white")
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Colunas de popularidade não disponíveis para o gráfico de dispersão.")


# ---------------------------------
# ABA: MAPA GLOBAL
# ---------------------------------
with aba_mapa:
    st.markdown("### 🌍 Distribuição global de títulos por país de produção")

    if "pais_nome" not in df_filt.columns:
        st.info("Os dados atuais não possuem a coluna `pais_nome` (execute o notebook PRO para enriquecer com países).")
    else:
        df_country = (
            df_filt.dropna(subset=["pais_nome"])
            .groupby("pais_nome", as_index=False)
            .agg(
                quantidade=("id", "count"),
                nota_media=("vote_average", "mean"),
                popularidade_media=("popularity", "mean"),
                lucro_medio=("lucro", "mean") if "lucro" in df_filt.columns else ("id", "count"),
            )
        )

        metrica = st.selectbox(
            "Métrica para o mapa",
            options=["quantidade", "nota_media", "popularidade_media", "lucro_medio"],
            format_func=lambda x: {
                "quantidade": "Quantidade de títulos",
                "nota_media": "Nota média",
                "popularidade_media": "Popularidade média",
                "lucro_medio": "Lucro médio",
            }.get(x, x),
        )

        fig_geo = px.choropleth(
            df_country,
            locations="pais_nome",
            locationmode="country names",
            color=metrica,
            hover_name="pais_nome",
            color_continuous_scale="Viridis",
            title=f"Mapa mundial – {metrica.replace('_', ' ').title()} por país",
        )
        fig_geo.update_layout(
            geo=dict(showframe=False, projection_type="natural earth"),
            template="plotly_white",
        )
        st.plotly_chart(fig_geo, use_container_width=True)

        st.markdown("#### 📊 Top 10 países (ordenado pela métrica selecionada)")
        st.dataframe(
            df_country.sort_values(metrica, ascending=False).head(10),
            use_container_width=True,
        )


# ---------------------------------
# ABA: GÊNEROS
# ---------------------------------
with aba_generos:
    st.markdown("### 🎭 Distribuição de Gêneros")

    if "generos_nomes" not in df_filt.columns:
        st.info("O dataframe não possui a coluna `generos_nomes`.")
    else:
        exploded_gen = df_filt[["tipo", "generos_nomes"]].explode("generos_nomes")
        exploded_gen = exploded_gen.dropna(subset=["generos_nomes"])

        if exploded_gen.empty:
            st.info("Nenhum gênero disponível para análise.")
        else:
            freq_gen = (
                exploded_gen.groupby("generos_nomes")
                .size()
                .reset_index(name="quantidade")
            )

            fig_gen = px.bar(
                freq_gen.sort_values("quantidade", ascending=True),
                x="quantidade",
                y="generos_nomes",
                orientation="h",
                title="Frequência dos gêneros (dados filtrados)",
                text="quantidade",
                color="quantidade",
                color_continuous_scale="Blues",
            )
            fig_gen.update_layout(
                template="plotly_white",
                xaxis_title="Quantidade",
                yaxis_title="Gênero",
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_gen, use_container_width=True)

            st.markdown("#### Gêneros por tipo de título")
            gen_tipo = (
                exploded_gen.groupby(["tipo", "generos_nomes"])
                .size()
                .reset_index(name="quantidade")
            )

            fig_gen_tipo = px.bar(
                gen_tipo,
                x="generos_nomes",
                y="quantidade",
                color="tipo",
                title="Gêneros por tipo de título",
            )
            fig_gen_tipo.update_layout(
                template="plotly_white",
                xaxis_title="Gênero",
                yaxis_title="Quantidade",
            )
            st.plotly_chart(fig_gen_tipo, use_container_width=True)


# ---------------------------------
# ABA: TOP TÍTULOS
# ---------------------------------
with aba_top:
    st.markdown("### 🏆 Top Títulos por Popularidade")

    cols_top = ["titulo_unico", "tipo", "ano_lancamento", "vote_average", "popularity"]
    cols_top = [c for c in cols_top if c in df_filt.columns]

    top_n = st.slider("Quantidade de títulos a exibir", 5, 50, 15, step=5)

    df_top = df_filt.sort_values("popularity", ascending=False)[cols_top].head(top_n)
    st.dataframe(df_top, use_container_width=True)

    if "titulo_unico" in df_top.columns:
        st.markdown("#### Destaques")
        for _, row in df_top.head(5).iterrows():
            st.write(
                f"**{row.get('titulo_unico', 'N/D')}** "
                f"({int(row.get('ano_lancamento')) if not np.isnan(row.get('ano_lancamento', np.nan)) else 's/ano'}) "
                f"- Tipo: {row.get('tipo', 'N/D')} | "
                f"Nota: {row.get('vote_average', 0):.1f} | "
                f"Popularidade: {row.get('popularity', 0):.1f}"
            )


# ---------------------------------
# ABA: CORRELAÇÃO
# ---------------------------------
with aba_corr:
    st.markdown("### 📈 Correlação entre variáveis numéricas")

    num_cols = df_filt.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if len(num_cols) < 2:
        st.info("Não há variáveis numéricas suficientes para calcular correlação.")
    else:
        corr = df_filt[num_cols].corr()

        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Mapa de correlação (dados filtrados)",
        )
        fig_corr.update_layout(template="plotly_white")
        st.plotly_chart(fig_corr, use_container_width=True)


# ---------------------------------
# ABA: TABELA + DOWNLOAD
# ---------------------------------
with aba_tabela:
    st.markdown("### 📄 Tabela de Títulos Filtrados")

    colunas_visiveis = [
        col
        for col in [
            "titulo_unico",
            "tipo",
            "ano_lancamento",
            "vote_average",
            "popularity",
            "tamanho_sinopse",
            "pais_nome",
            "generos_str",
        ]
        if col in df_filt.columns
    ]

    st.dataframe(df_filt[colunas_visiveis], use_container_width=True)

    csv_download = df_filt[colunas_visiveis].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Baixar CSV filtrado",
        data=csv_download,
        file_name="tmdb_titulos_filtrados.csv",
        mime="text/csv",
    )

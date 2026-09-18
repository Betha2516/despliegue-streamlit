# -*- coding: utf-8 -*-
"""
Despliegue videojuegos - App de predicción de inversión en tienda de videojuegos
Versión con interfaz mejorada: layout en dos columnas, barra lateral para
captura de datos, tarjetas con sombra, colores y micro-interacciones.
"""

import pickle
import pandas as pd
import streamlit as st

# ============================================================
# Configuración general de la página
# ============================================================
st.set_page_config(
    page_title="Predicción de Inversión | Videojuegos",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Estilos personalizados (CSS)
# ============================================================
st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #C026D3 100%);
        padding: 2.2rem 2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 24px rgba(124, 58, 237, 0.25);
    }
    .hero h1 { margin: 0; font-size: 2rem; }
    .hero p  { margin: 0.4rem 0 0 0; opacity: 0.9; font-size: 1.05rem; }

    .section-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        border: 1px solid #f0f0f0;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #eee;
        border-radius: 14px;
        padding: 0.8rem 1rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
    }

    .result-box {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
    }
    .result-box .label { opacity: 0.9; font-size: 1rem; }
    .result-box .value { font-size: 2.6rem; font-weight: 700; margin-top: 0.2rem; }

    section[data-testid="stSidebar"] { background: #FAFAFF; }

    .stButton > button {
        background: linear-gradient(135deg, #7C3AED, #C026D3);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover { opacity: 0.92; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Carga del modelo (con caché para no recargarlo en cada interacción)
# ============================================================
@st.cache_resource
def cargar_modelo():
    with open("modelo-reg.pkl", "rb") as f:
        modelo, min_max_scaler, variables = pickle.load(f)
    return modelo, min_max_scaler, variables


modelo, min_max_scaler, variables = cargar_modelo()

# ============================================================
# Encabezado principal
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>🎮 Predicción de Inversión en Videojuegos</h1>
        <p>Estima cuánto invertirá un cliente en la tienda según su perfil de consumo</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Barra lateral: captura de datos
# ============================================================
with st.sidebar:
    st.markdown("### 📋 Perfil del cliente")
    st.caption("Completa los datos y presiona **Calcular** para generar la predicción.")

    Edad = st.slider("Edad", min_value=14, max_value=52, value=20, step=1)

    videojuegos_opciones = [
        "'Mass Effect'", "'Battlefield'", "'Fifa'", "'KOA: Reckoning'",
        "'Crysis'", "'Sim City'", "'Dead Space'", "'F1'",
    ]
    videojuego = st.selectbox(
        "🕹️ Videojuego favorito",
        videojuegos_opciones,
        format_func=lambda x: x.strip("'"),
    )

    plataforma_opciones = ["'Play Station'", "'Xbox'", "PC", "Otros"]
    Plataforma = st.selectbox(
        "🖥️ Plataforma",
        plataforma_opciones,
        format_func=lambda x: x.strip("'"),
    )

    Sexo = st.radio("👤 Sexo", ["Hombre", "Mujer"], horizontal=True)

    Consumidor_habitual = st.selectbox(
        "🛒 ¿Consumidor habitual?",
        ["True", "False"],
        format_func=lambda x: "Sí" if x == "True" else "No",
    )

    st.markdown("---")
    calcular = st.button("🔮 Calcular predicción", type="primary", use_container_width=True)

# Versiones "limpias" (sin comillas) solo para mostrar en pantalla
videojuego_display = videojuego.strip("'")
plataforma_display = Plataforma.strip("'")

# ============================================================
# Preparación de datos (misma lógica del modelo original)
# ============================================================
datos = [[Edad, videojuego, Plataforma, Sexo, Consumidor_habitual]]
data = pd.DataFrame(
    datos,
    columns=["Edad", "videojuego", "Plataforma", "Sexo", "Consumidor_habitual"],
)

data_preparada = data.copy()
data_preparada = pd.get_dummies(
    data_preparada,
    columns=["videojuego", "Plataforma", "Sexo", "Consumidor_habitual"],
    drop_first=False,
    dtype=int,
)
data_preparada = data_preparada.reindex(columns=variables, fill_value=0)

Y_pred = modelo.predict(data_preparada)
data["Prediccion"] = Y_pred

# ============================================================
# Cuerpo principal: resumen del perfil + resultado
# ============================================================
col_izq, col_der = st.columns([1, 1.2], gap="large")

with col_izq:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### 👤 Resumen del perfil")

    c1, c2 = st.columns(2)
    c1.metric("Edad", f"{Edad} años")
    c2.metric("Sexo", Sexo)

    c3, c4 = st.columns(2)
    c3.metric("Plataforma", plataforma_display)
    c4.metric("Consumidor habitual", "Sí" if Consumidor_habitual == "True" else "No")

    st.markdown(f"🕹️ **Videojuego favorito:** {videojuego_display}")
    st.markdown("</div>", unsafe_allow_html=True)

with col_der:
    if calcular:
        prediccion = float(Y_pred[0])
        st.markdown(
            f"""
            <div class="result-box">
                <div class="label">💰 Inversión estimada</div>
                <div class="value">${prediccion:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info("📊 El modelo tiene un error del **16 %** (MAPE: error porcentual absoluto medio).")
    else:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### 🔮 Resultado")
        st.info(
            "Completa el perfil en la barra lateral y presiona **Calcular predicción** "
            "para ver la estimación."
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Detalles técnicos (opcional, colapsado por defecto)
# ============================================================
with st.expander("🔧 Ver detalles técnicos"):
    st.dataframe(data, use_container_width=True)

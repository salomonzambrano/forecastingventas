import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Simulador de Ventas Nov 2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
        border: none;
        font-size: 18px;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    h1, h2, h3 {
        color: white;
    }
    .dataframe {
        background-color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Función para cargar datos
@st.cache_resource
def cargar_modelo_y_datos():
    try:
        modelo = joblib.load('/models/modelo_final.joblib')
        df = pd.read_csv('../data/processed/inferencia_df_transformado.csv')
        df['fecha'] = pd.to_datetime(df['fecha'])
        return modelo, df
    except Exception as e:
        st.error(f"❌ Error al cargar datos: {str(e)}")
        return None, None

# Función para predicciones recursivas
def predecir_recursivamente(df_producto, modelo, ajuste_descuento, ajuste_competencia):
    """
    Realiza predicciones día por día actualizando los lags recursivamente
    """
    df_pred = df_producto.copy().sort_values('fecha').reset_index(drop=True)
    predicciones = []
    
    # Obtener features que el modelo espera
    features_modelo = modelo.feature_names_in_
    
    for i in range(len(df_pred)):
        # Calcular precio_venta con el ajuste de descuento
        precio_base = df_pred.loc[i, 'precio_base']
        descuento = ajuste_descuento / 100
        precio_venta = precio_base * (1 + descuento)
        df_pred.loc[i, 'precio_venta'] = precio_venta
        df_pred.loc[i, 'descuento_porcentaje'] = descuento * 100
        
        # Calcular precio_competencia con el ajuste
        precio_comp_original = df_pred.loc[i, 'precio_competencia']
        precio_comp_ajustado = precio_comp_original * (1 + ajuste_competencia / 100)
        df_pred.loc[i, 'precio_competencia'] = precio_comp_ajustado
        
        # Recalcular ratio_precio
        df_pred.loc[i, 'ratio_precio'] = precio_venta / precio_comp_ajustado if precio_comp_ajustado > 0 else 1.0
        
        # Preparar datos para predicción
        X = df_pred.loc[[i], features_modelo]
        
        # Predecir
        pred = modelo.predict(X)[0]
        pred = max(0, pred)  # No permitir predicciones negativas
        predicciones.append(pred)
        
        # Actualizar lags para el siguiente día (si no es el último)
        if i < len(df_pred) - 1:
            # Desplazar lags hacia la derecha
            for lag in range(7, 1, -1):
                col_actual = f'unidades_vendidas_lag{lag}'
                col_anterior = f'unidades_vendidas_lag{lag-1}'
                if col_anterior in df_pred.columns:
                    df_pred.loc[i+1, col_actual] = df_pred.loc[i, col_anterior]
            
            # Actualizar lag_1 con la predicción actual
            df_pred.loc[i+1, 'unidades_vendidas_lag1'] = pred
            
            # Actualizar media móvil de 7 días
            ultimas_7 = predicciones[-7:] if len(predicciones) >= 7 else predicciones
            df_pred.loc[i+1, 'unidades_vendidas_mm7'] = np.mean(ultimas_7)
    
    df_pred['unidades_predichas'] = predicciones
    df_pred['ingresos_proyectados'] = df_pred['unidades_predichas'] * df_pred['precio_venta']
    
    return df_pred

# Cargar modelo y datos
modelo, df_completo = cargar_modelo_y_datos()

if modelo is None or df_completo is None:
    st.stop()

# SIDEBAR - Controles de Simulación
st.sidebar.title("🎛️ Controles de Simulación")
st.sidebar.markdown("---")

# Selector de producto
productos = sorted(df_completo['nombre'].unique())
producto_seleccionado = st.sidebar.selectbox(
    "📦 Selecciona un producto:",
    productos,
    index=0
)

# Slider de descuento
ajuste_descuento = st.sidebar.slider(
    "💰 Ajuste de descuento (%)",
    min_value=-50,
    max_value=50,
    value=0,
    step=5,
    help="Ajusta el descuento sobre el precio base"
)

# Selector de escenario de competencia
st.sidebar.markdown("### 🏪 Escenario de Competencia")
escenario_competencia = st.sidebar.radio(
    "Selecciona el escenario:",
    ["Actual (0%)", "Competencia -5%", "Competencia +5%"],
    index=0
)

# Mapear escenario a valor numérico
ajuste_comp_map = {
    "Actual (0%)": 0,
    "Competencia -5%": -5,
    "Competencia +5%": 5
}
ajuste_competencia = ajuste_comp_map[escenario_competencia]

st.sidebar.markdown("---")

# Botón de simulación
simular = st.sidebar.button("🚀 Simular Ventas")

# ZONA PRINCIPAL
if simular:
    with st.spinner('🔮 Generando predicciones recursivas...'):
        # Filtrar datos del producto seleccionado
        df_producto = df_completo[df_completo['nombre'] == producto_seleccionado].copy()
        
        if df_producto.empty:
            st.error("❌ No hay datos para el producto seleccionado")
            st.stop()
        
        # Realizar predicciones recursivas
        df_resultado = predecir_recursivamente(
            df_producto, 
            modelo, 
            ajuste_descuento, 
            ajuste_competencia
        )
        
        # HEADER
        st.markdown(f"# 📊 Dashboard de Simulación - Noviembre 2025")
        st.markdown(f"### 🏷️ Producto: **{producto_seleccionado}**")
        st.markdown("---")
        
        # KPIs DESTACADOS
        col1, col2, col3, col4 = st.columns(4)
        
        unidades_totales = df_resultado['unidades_predichas'].sum()
        ingresos_totales = df_resultado['ingresos_proyectados'].sum()
        precio_promedio = df_resultado['precio_venta'].mean()
        descuento_promedio = df_resultado['descuento_porcentaje'].mean()
        
        with col1:
            st.metric(
                "📦 Unidades Totales",
                f"{int(unidades_totales):,}",
                help="Total de unidades proyectadas para noviembre"
            )
        
        with col2:
            st.metric(
                "💰 Ingresos Totales",
                f"€{ingresos_totales:,.2f}",
                help="Ingresos totales proyectados"
            )
        
        with col3:
            st.metric(
                "🏷️ Precio Promedio",
                f"€{precio_promedio:.2f}",
                help="Precio promedio de venta"
            )
        
        with col4:
            st.metric(
                "🎯 Descuento Promedio",
                f"{descuento_promedio:.1f}%",
                help="Descuento promedio aplicado"
            )
        
        st.markdown("---")
        
        # GRÁFICO DE PREDICCIÓN DIARIA
        st.markdown("## 📈 Predicción Diaria de Ventas")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Configurar estilo
        sns.set_style("whitegrid")
        
        # Crear el gráfico de línea
        dias = df_resultado['dia_mes'].values
        ventas = df_resultado['unidades_predichas'].values
        
        ax.plot(dias, ventas, linewidth=2.5, color='#667eea', marker='o', 
                markersize=6, label='Ventas Predichas')
        
        # Marcar Black Friday (día 28)
        idx_bf = df_resultado[df_resultado['dia_mes'] == 28].index[0]
        dia_bf = df_resultado.loc[idx_bf, 'dia_mes']
        venta_bf = df_resultado.loc[idx_bf, 'unidades_predichas']
        
        ax.axvline(x=dia_bf, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.plot(dia_bf, venta_bf, 'ro', markersize=15, zorder=5)
        ax.annotate('🛍️ BLACK FRIDAY', 
                   xy=(dia_bf, venta_bf), 
                   xytext=(dia_bf-5, venta_bf*1.15),
                   fontsize=12, 
                   fontweight='bold',
                   color='red',
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        ax.set_xlabel('Día de Noviembre', fontsize=12, fontweight='bold')
        ax.set_ylabel('Unidades Vendidas', fontsize=12, fontweight='bold')
        ax.set_title('Predicción de Ventas - Noviembre 2025', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 31)
        
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        # TABLA DETALLADA
        st.markdown("## 📋 Detalle Diario de Predicciones")
        
        # Preparar tabla para mostrar
        df_tabla = df_resultado[['fecha', 'nombre_dia_semana', 'precio_venta', 
                                  'precio_competencia', 'descuento_porcentaje', 
                                  'unidades_predichas', 'ingresos_proyectados']].copy()
        
        df_tabla['fecha'] = df_tabla['fecha'].dt.strftime('%Y-%m-%d')
        df_tabla['precio_venta'] = df_tabla['precio_venta'].apply(lambda x: f"€{x:.2f}")
        df_tabla['precio_competencia'] = df_tabla['precio_competencia'].apply(lambda x: f"€{x:.2f}")
        df_tabla['descuento_porcentaje'] = df_tabla['descuento_porcentaje'].apply(lambda x: f"{x:.1f}%")
        df_tabla['unidades_predichas'] = df_tabla['unidades_predichas'].apply(lambda x: f"{int(x)}")
        df_tabla['ingresos_proyectados'] = df_tabla['ingresos_proyectados'].apply(lambda x: f"€{x:.2f}")
        
        # Añadir emoji a Black Friday
        df_tabla['fecha'] = df_tabla.apply(
            lambda row: f"🛍️ {row['fecha']}" if '28' in row['fecha'] else row['fecha'],
            axis=1
        )
        
        df_tabla.columns = ['Fecha', 'Día Semana', 'Precio Venta', 'Precio Competencia', 
                           'Descuento', 'Unidades', 'Ingresos']
        
        st.dataframe(df_tabla, use_container_width=True, height=400)
        
        st.markdown("---")
        
        # COMPARATIVA DE ESCENARIOS
        st.markdown("## 🔄 Comparativa de Escenarios de Competencia")
        st.markdown("_Manteniendo el descuento actual y variando solo los precios de la competencia_")
        
        escenarios = [
            ("Competencia Actual (0%)", 0),
            ("Competencia Reducida (-5%)", -5),
            ("Competencia Aumentada (+5%)", 5)
        ]
        
        resultados_escenarios = []
        
        with st.spinner('⚙️ Calculando escenarios alternativos...'):
            for nombre_esc, ajuste_esc in escenarios:
                df_esc = predecir_recursivamente(
                    df_producto,
                    modelo,
                    ajuste_descuento,
                    ajuste_esc
                )
                resultados_escenarios.append({
                    'escenario': nombre_esc,
                    'unidades': df_esc['unidades_predichas'].sum(),
                    'ingresos': df_esc['ingresos_proyectados'].sum()
                })
        
        col1, col2, col3 = st.columns(3)
        
        for i, (col, resultado) in enumerate(zip([col1, col2, col3], resultados_escenarios)):
            with col:
                # Destacar el escenario seleccionado
                emoji = "⭐" if resultado['escenario'].split()[1].strip('()') in escenario_competencia else "📊"
                st.markdown(f"### {emoji} {resultado['escenario']}")
                st.metric(
                    "Unidades Totales",
                    f"{int(resultado['unidades']):,}"
                )
                st.metric(
                    "Ingresos Totales",
                    f"€{resultado['ingresos']:,.2f}"
                )
        
        st.markdown("---")
        st.success("✅ Simulación completada exitosamente")
        
else:
    # Pantalla inicial
    st.markdown("# 📊 Simulador de Ventas - Noviembre 2025")
    st.markdown("---")
    st.info("👈 Configura los parámetros en el panel lateral y presiona **'🚀 Simular Ventas'** para comenzar")
    
    # Mostrar información del dataset
    st.markdown("## 📦 Productos Disponibles")
    st.markdown(f"**Total de productos:** {len(productos)}")
    
    # Mostrar categorías
    categorias = df_completo.groupby('categoria')['nombre'].count().reset_index()
    categorias.columns = ['Categoría', 'Cantidad de Productos']
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(categorias, use_container_width=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=categorias, x='Cantidad de Productos', y='Categoría', 
                   palette='viridis', ax=ax)
        ax.set_title('Distribución de Productos por Categoría', fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    st.markdown("### 🎯 Características del Simulador")
    st.markdown("""
    - **Predicciones Recursivas**: Los lags se actualizan día a día con las predicciones previas
    - **Escenarios de Competencia**: Simula cambios en los precios de la competencia
    - **Ajuste de Descuentos**: Modifica el descuento aplicado sobre el precio base
    - **Visualización Completa**: Gráficos interactivos y tablas detalladas
    - **Black Friday Destacado**: Identifica automáticamente el 28 de noviembre
    """)
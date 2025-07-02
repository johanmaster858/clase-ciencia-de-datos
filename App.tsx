
import React from 'react';
import CodeBlock from './components/CodeBlock';
import SectionCard from './components/SectionCard';
import AnalysisBlock from './components/AnalysisBlock';
import HousingHistogram from './components/charts/HousingHistogram';
import CorrelationHeatmap from './components/charts/CorrelationHeatmap';
import IncomeScatterPlot from './components/charts/IncomeScatterPlot';
import ClusterMap from './components/charts/ClusterMap';
import WineQualityHistogram from './components/charts/WineQualityHistogram';
import WineCorrelationHeatmap from './components/charts/WineCorrelationHeatmap';
import WineClassifier from './components/WineClassifier';

const App: React.FC = () => {

  const pythonCode = {
    loadLibraries: `
# Librería para manipulación de datos
import pandas as pd
# Librería para operaciones numéricas
import numpy as np
# Librerías para visualización
import matplotlib.pyplot as plt
import seaborn as sns
# Librería para Machine Learning
from sklearn.datasets import fetch_california_housing

print("Librerías cargadas exitosamente!")`,
    loadDataset: `
# Cargamos el dataset desde sklearn
housing = fetch_california_housing()

# Para facilitar su manipulación, lo convertimos a un DataFrame de Pandas
# housing.data contiene las características (features)
# housing.feature_names contiene los nombres de las columnas
# housing.target contiene el valor a predecir (el precio de la vivienda)
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# Añadimos la columna objetivo (el precio) a nuestro DataFrame
# El precio está en unidades de 100,000 USD
df['MedHouseVal'] = housing.target

print("Dataset cargado. Aquí tienes una muestra:")
df.head()`,
    exploreData: `
# Obtener información general del DataFrame: tipos de datos y valores no nulos
print("Información general del dataset:")
df.info()

print("\\n----------------------------------\\n")

# Comprobar si hay valores nulos en alguna columna
print("Suma de valores nulos por columna:")
print(df.isnull().sum())

print("\\n----------------------------------\\n")

# Obtener estadísticas descriptivas de las columnas numéricas
print("Estadísticas descriptivas:")
df.describe()`,
    histogramPlot: `
# Histograma de todas las variables numéricas para ver su distribución
df.hist(bins=50, figsize=(20, 15))
plt.suptitle("Distribución de las Características del Dataset", size=20)
plt.show()`,
    correlationPlot: `
# Matriz de correlación para entender la relación entre variables
# La correlación mide cómo dos variables se mueven juntas (-1 a 1)
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.show()`,
    scatterPlot: `
# Scatter plot (diagrama de dispersión) para visualizar la relación más importante
df.plot(kind="scatter", x="MedInc", y="MedHouseVal", alpha=0.2,
        figsize=(10,7), c="MedHouseVal", cmap=plt.get_cmap("jet"), colorbar=True,
        )
plt.title("Precio de la Vivienda vs. Ingreso Medio")
plt.xlabel("Ingreso Medio (en decenas de miles de $)")
plt.ylabel("Valor Medio de la Vivienda (en cientos de miles de $)")
plt.show()`,
    prepareData: `
from sklearn.model_selection import train_test_split

# X contiene todas las columnas EXCEPTO el precio
X = df.drop("MedHouseVal", axis=1)
# y contiene SOLAMENTE la columna del precio
y = df["MedHouseVal"]

# Dividimos los datos: 80% para entrenar, 20% para probar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Tamaño del set de entrenamiento: {X_train.shape[0]} filas")
print(f"Tamaño del set de prueba: {X_test.shape[0]} filas")`,
    trainModel: `
from sklearn.ensemble import RandomForestRegressor

# Creamos una instancia del modelo
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)

# Entrenamos el modelo con nuestros datos de entrenamiento
forest_reg.fit(X_train, y_train)

print("¡Modelo de Random Forest entrenado!")`,
    evaluateModel: `
from sklearn.metrics import mean_squared_error

# Hacemos predicciones sobre el conjunto de prueba
predictions = forest_reg.predict(X_test)

# Calculamos el error
forest_mse = mean_squared_error(y_test, predictions)
forest_rmse = np.sqrt(forest_mse)

print(f"El Error Cuadrático Medio Raíz (RMSE) del modelo es: ${'{'}forest_rmse * 100000:.2f{'}'}")`,
    kmeansCluster: `
from sklearn.cluster import KMeans

# Seleccionamos solo las coordenadas geográficas
X_cluster = df[['Latitude', 'Longitude']]

# Creamos y entrenamos el modelo K-Means, pidiéndole que encuentre 6 grupos (clusters)
kmeans = KMeans(n_clusters=6, random_state=42, n_init='auto')
df['Cluster'] = kmeans.fit_predict(X_cluster)

print("Se han asignado los clusters. Veamos los primeros registros con su cluster:")
print(df.head())

# Visualicemos los clusters en un mapa
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='Longitude', y='Latitude', hue='Cluster', palette='viridis', alpha=0.5)
plt.title('Clusters de Vecindarios en California por Ubicación')
plt.show()`,
    wine_loadDataset: `
# Nota: Usaremos el dataset de vino tinto.
# El dataset se puede encontrar en el Repositorio de Machine Learning de UCI.
import pandas as pd
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

# Usamos ';' como separador porque el CSV original lo usa.
wine_df = pd.read_csv(url, sep=';')

print("Dataset de calidad de vino tinto cargado. Aquí tienes una muestra:")
wine_df.head()`,
    wine_exploreData: `
# Obtener información general del dataset: tipos de datos y valores no nulos
print("Información general del dataset de vinos:")
wine_df.info()

print("\\n----------------------------------\\n")

# Comprobar si hay valores nulos en alguna columna
print("Suma de valores nulos por columna:")
print(wine_df.isnull().sum())

print("\\n----------------------------------\\n")

# Obtener estadísticas descriptivas de las columnas numéricas
print("Estadísticas descriptivas:")
wine_df.describe()`,
    wine_qualityHistogram: `
import matplotlib.pyplot as plt
import seaborn as sns

# Histograma para ver la distribución de la variable 'quality'
plt.figure(figsize=(8, 5))
sns.countplot(x='quality', data=wine_df, palette='viridis')
plt.title('Distribución de la Calidad del Vino')
plt.xlabel('Calidad (Puntuación)')
plt.ylabel('Número de Vinos')
plt.show()`,
    wine_correlationPlot: `
# Matriz de correlación para entender la relación entre variables
plt.figure(figsize=(12, 10))
correlation_matrix_wine = wine_df.corr()
sns.heatmap(correlation_matrix_wine, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlación de las Características del Vino')
plt.show()`,
    wine_prepareData: `
from sklearn.model_selection import train_test_split

# Para este problema, convertiremos la puntuación de calidad en una categoría binaria.
# Un vino con calidad >= 7 será 'bueno' (1), y el resto 'malo' (0).
# Esto transforma un problema de regresión en uno de clasificación, que es muy común.
wine_df['quality_label'] = wine_df['quality'].apply(lambda value: 1 if value >= 7 else 0)

# X contiene las características, y es la variable objetivo.
X = wine_df.drop(['quality', 'quality_label'], axis=1)
y = wine_df['quality_label']

# Dividimos los datos: 80% para entrenar, 20% para probar
# Usamos 'stratify=y' para asegurar que la proporción de vinos buenos/malos es la misma en ambos sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Set de entrenamiento: {X_train.shape[0]} vinos")
print(f"Set de prueba: {X_test.shape[0]} vinos")
print("\\nDistribución de clases en el set de prueba:")
print(y_test.value_counts())`,
    wine_trainModel: `
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Es una buena práctica escalar las características para la Regresión Logística.
# Usamos un 'pipeline' para encadenar el escalado y el entrenamiento del modelo.
pipeline = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))

# Entrenamos el pipeline completo con los datos de entrenamiento
pipeline.fit(X_train, y_train)

print("¡Modelo de clasificación de vinos entrenado!")`,
    wine_evaluateModel: `
from sklearn.metrics import accuracy_score, classification_report

# Hacemos predicciones sobre el conjunto de prueba
predictions = pipeline.predict(X_test)

# Calculamos la precisión (accuracy)
accuracy = accuracy_score(y_test, predictions)

print(f"La precisión del modelo en la clasificación de vinos es: {accuracy:.2%}")

# El reporte de clasificación nos da más detalles (precisión, recall, f1-score)
print("\\nReporte de Clasificación detallado:")
print(classification_report(y_test, predictions, target_names=['Vino Malo', 'Vino Bueno']))`,
    wine_predictNew: `
# Ahora, usemos nuestro pipeline para predecir la calidad de un vino hipotético.
# Creamos un DataFrame con una sola fila con los nuevos datos.
# ¡Asegúrate de que el orden y los nombres de las columnas coincidan con los datos de entrenamiento!
nuevo_vino = pd.DataFrame({
    'fixed acidity': [7.8], 'volatile acidity': [0.58], 'citric acid': [0.02],
    'residual sugar': [2.0], 'chlorides': [0.073], 'free sulfur dioxide': [9.0],
    'total sulfur dioxide': [18.0], 'density': [0.9968], 'pH': [3.36],
    'sulphates': [0.57], 'alcohol': [11.2]
})

# Usamos el pipeline para predecir.
prediccion = pipeline.predict(nuevo_vino)
probabilidades = pipeline.predict_proba(nuevo_vino)

# Imprimimos el resultado.
resultado = "Buena Calidad" if prediccion[0] == 1 else "Calidad Regular"
confianza = probabilidades[0][prediccion[0]] * 100

print(f"El modelo predice que este es un vino de: {resultado}")
print(f"Confianza de la predicción: {confianza:.2f}%")`,
    wine_interactive_ui: `
import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd

# Asumimos que el 'pipeline' y 'X.columns' de los pasos anteriores ya están disponibles.

# Diccionario de sliders para cada característica
sliders = {
    'fixed acidity': widgets.FloatSlider(value=8.3, min=4.6, max=15.9, step=0.1, description='Acidez Fija', readout_format='.1f'),
    'volatile acidity': widgets.FloatSlider(value=0.53, min=0.12, max=1.58, step=0.01, description='Acidez Volátil', readout_format='.2f'),
    'citric acid': widgets.FloatSlider(value=0.27, min=0.0, max=1.0, step=0.01, description='Ácido Cítrico', readout_format='.2f'),
    'residual sugar': widgets.FloatSlider(value=2.5, min=0.9, max=15.5, step=0.1, description='Azúcar Residual', readout_format='.1f'),
    'chlorides': widgets.FloatSlider(value=0.087, min=0.012, max=0.611, step=0.001, description='Cloruros', readout_format='.3f'),
    'free sulfur dioxide': widgets.FloatSlider(value=16.0, min=1.0, max=72.0, step=1, description='SO2 Libre', readout_format='.0f'),
    'total sulfur dioxide': widgets.FloatSlider(value=46.0, min=6.0, max=289.0, step=1, description='SO2 Total', readout_format='.0f'),
    'density': widgets.FloatSlider(value=0.9967, min=0.99, max=1.004, step=0.00001, description='Densidad', readout_format='.5f'),
    'pH': widgets.FloatSlider(value=3.31, min=2.74, max=4.01, step=0.01, description='pH', readout_format='.2f'),
    'sulphates': widgets.FloatSlider(value=0.66, min=0.33, max=2.0, step=0.01, description='Sulfatos', readout_format='.2f'),
    'alcohol': widgets.FloatSlider(value=10.4, min=8.4, max=14.9, step=0.1, description='Alcohol', readout_format='.1f'),
}

# Botón y área de salida
button = widgets.Button(description="Clasificar Vino", button_style='success')
output = widgets.Output()

def classify_wine(b):
    with output:
        clear_output()
        input_data = {name: [slider.value] for name, slider in sliders.items()}
        # Aseguramos el orden correcto de las columnas
        nuevo_vino_df = pd.DataFrame(input_data, columns=X.columns)
        
        prediction = pipeline.predict(nuevo_vino_df)
        proba = pipeline.predict_proba(nuevo_vino_df)
        
        resultado = "Buena Calidad" if prediction[0] == 1 else "Calidad Regular"
        confianza = proba[0][prediction[0]] * 100
        
        print(f"Predicción: {resultado} (Confianza: {confianza:.2f}%)")

button.on_click(classify_wine)

# Muestra la UI organizada verticalmente
print("### Clasificador de Vino Interactivo para Colab ###")
ui = widgets.VBox(list(sliders.values()) + [button, output])
display(ui)`
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-700">
      <main className="container mx-auto px-4 py-8 md:py-16">
        
        <header className="text-center mb-16">
          <h1 className="text-4xl md:text-6xl font-extrabold text-slate-900 mb-4 tracking-tight">
            <span className="block">Masterclass de Ciencia de Datos:</span>
            <span className="block text-indigo-600">De los Conceptos a la Práctica con Python</span>
          </h1>
          <p className="max-w-3xl mx-auto text-lg md:text-xl text-slate-600">
            ¡Bienvenido/a al mundo de los datos! En esta clase magistral intensiva, desmitificaremos la ciencia de datos y te daremos las herramientas para que empieces a transformar información en conocimiento accionable.
          </p>
        </header>

        <SectionCard title="Parte 1: La Aventura de la Ciencia de Datos" subtitle="Introducción">
          <h3 className="text-2xl font-bold text-slate-800 mt-4 mb-3">¿Qué es realmente la Ciencia de Datos?</h3>
          <p className="mb-4">Imagina que eres un detective. Los datos son tus pistas: registros de ventas, interacciones en redes sociales, lecturas de sensores, textos, imágenes... La ciencia de datos es el conjunto de técnicas y herramientas que te permiten interrogar esas pistas, encontrar patrones ocultos, resolver misterios y predecir lo que podría suceder a continuación.</p>
          <p className="mb-4">En esencia, es una disciplina que combina estadística, informática y conocimiento de negocio para extraer valor de los datos y ayudar a tomar mejores decisiones.</p>
          <p className="mb-6 italic">El trabajo no es un evento único, sino un ciclo de vida continuo: comienza con una pregunta de negocio, seguido por la recolección y limpieza de datos, el análisis exploratorio para encontrar pistas, la construcción de modelos predictivos y, finalmente, la comunicación de los hallazgos a quienes toman las decisiones.</p>
          
          <h3 className="text-2xl font-bold text-slate-800 mt-8 mb-3">Habilidades Clave del Científico de Datos Moderno</h3>
          <ul className="list-disc list-inside space-y-2 mb-6">
              <li><strong>Programación (Python es el rey):</strong> Para manipular datos y construir modelos.</li>
              <li><strong>Estadística y Matemáticas:</strong> Para entender los patrones, la incertidumbre y la validez de los resultados.</li>
              <li><strong>Limpieza y Manipulación de Datos (Data Wrangling):</strong> El 80% del trabajo. Los datos del mundo real son desordenados y necesitan ser preparados.</li>
              <li><strong>Visualización de Datos y Comunicación:</strong> Un hallazgo no sirve de nada si no puedes explicarlo. Contar historias con datos es fundamental.</li>
              <li><strong>Machine Learning:</strong> Para construir modelos que aprenden de los datos para hacer predicciones o clasificaciones.</li>
              <li><strong>Curiosidad y Pensamiento Crítico:</strong> La habilidad más importante. Siempre preguntar "¿por qué?" y "¿qué pasaría si...?".</li>
          </ul>

          <h3 className="text-2xl font-bold text-slate-800 mt-12 mb-3">Campos de Aplicación y Acción del Profesional</h3>
          <p className="mb-8">Los científicos de datos son versátiles y se encuentran en el corazón de la innovación en prácticamente todas las industrias. Aquí vemos algunos ejemplos clave:</p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
            <div className="bg-slate-100 p-6 rounded-lg shadow-sm ring-1 ring-slate-200">
              <img src="https://images.unsplash.com/photo-1554224155-8d04421cd6a8?q=80&w=800&auto=format&fit=crop" alt="Analista financiero trabajando en gráficos" className="w-full h-48 object-cover rounded-lg mb-4 shadow-md" />
              <h4 className="font-bold text-xl text-indigo-700 mb-2">Finanzas y Banca</h4>
              <p>Combaten el fraude analizando patrones de transacciones, desarrollan modelos de riesgo crediticio para decidir la viabilidad de un préstamo y crean algoritmos para el trading de alta frecuencia.</p>
            </div>
            
            <div className="bg-slate-100 p-6 rounded-lg shadow-sm ring-1 ring-slate-200">
              <img src="https://images.unsplash.com/photo-1576091160550-2173dba999ab?q=80&w=800&auto=format&fit=crop" alt="Doctor viendo una imagen médica en una tablet" className="w-full h-48 object-cover rounded-lg mb-4 shadow-md" />
              <h4 className="font-bold text-xl text-indigo-700 mb-2">Salud y Medicina</h4>
              <p>Ayudan a diagnosticar enfermedades mediante el análisis de imágenes médicas, aceleran el descubrimiento de nuevos fármacos analizando datos genómicos y desarrollan planes de tratamiento personalizados.</p>
            </div>

            <div className="bg-slate-100 p-6 rounded-lg shadow-sm ring-1 ring-slate-200">
              <img src="https://images.unsplash.com/photo-1522204523234-8729aa6e3d5f?q=80&w=800&auto=format&fit=crop" alt="Persona comprando en línea desde su laptop" className="w-full h-48 object-cover rounded-lg mb-4 shadow-md" />
              <h4 className="font-bold text-xl text-indigo-700 mb-2">E-commerce y Retail</h4>
              <p>Crean los motores de recomendación que te sugieren qué comprar, optimizan la cadena de suministro prediciendo la demanda y personalizan la experiencia del usuario con marketing dirigido.</p>
            </div>
            
            <div className="bg-slate-100 p-6 rounded-lg shadow-sm ring-1 ring-slate-200">
              <img src="https://images.unsplash.com/photo-1594904351111-a072f80b1a71?q=80&w=800&auto=format&fit=crop" alt="Interfaz de una plataforma de streaming de video" className="w-full h-48 object-cover rounded-lg mb-4 shadow-md" />
              <h4 className="font-bold text-xl text-indigo-700 mb-2">Entretenimiento y Medios</h4>
              <p>Son la magia detrás de las recomendaciones de Netflix y las playlists de Spotify. Analizan los hábitos de consumo para sugerir contenido nuevo y ayudar a los estudios a predecir qué guiones tendrán éxito.</p>
            </div>
          </div>
          
          <h3 className="text-2xl font-bold text-slate-800 mt-12 mb-3">¿Qué puedes lograr? El Poder de los Datos en Acción</h3>
           <p className="mb-4">Las posibilidades son casi infinitas. Empresas y organizaciones de todo el mundo usan la ciencia de datos para:</p>
          <ul className="list-disc list-inside space-y-2 mb-6 bg-slate-100 p-4 rounded-lg">
              <li>Predecir la demanda de un producto para optimizar el inventario.</li>
              <li>Detectar transacciones fraudulentas en tiempo real.</li>
              <li>Segmentar clientes para campañas de marketing personalizadas.</li>
              <li>Recomendarte la próxima película en Netflix o la siguiente canción en Spotify.</li>
              <li>Optimizar rutas de entrega para ahorrar tiempo y combustible.</li>
              <li>Ayudar a diagnosticar enfermedades a partir de imágenes médicas.</li>
          </ul>
          <p className="font-semibold text-indigo-700">En este taller, construiremos nuestro propio proyecto: predecir el valor de las viviendas en California basándonos en sus características.</p>
        </SectionCard>

        <SectionCard title="Parte 2: Taller Práctico" subtitle="Prediciendo el Valor de la Vivienda en Google Colab">
          <p className="mb-4">¡Manos a la obra! Abriremos Google Colab y construiremos un proyecto de ciencia de datos de principio a fin.</p>
          <div className="bg-indigo-50 border-l-4 border-indigo-500 text-indigo-800 p-4 rounded-r-lg mb-8">
            <h4 className="font-bold">¿Cómo empezar?</h4>
            <ol className="list-decimal list-inside mt-2">
              <li>Ve a <a href="https://colab.research.google.com/" target="_blank" rel="noopener noreferrer" className="font-medium underline">https://colab.research.google.com/</a>.</li>
              <li>Haz clic en <span className="font-mono bg-indigo-100 px-1 rounded">Archivo &gt; Nuevo cuaderno</span>.</li>
              <li>¡Listo! Ya tienes un entorno de Python listo para usar. Copia y pega los siguientes bloques de código en las celdas de tu cuaderno y ejecútalos (con <span className="font-mono bg-indigo-100 px-1 rounded">Shift + Enter</span>).</li>
            </ol>
          </div>
        </SectionCard>
        
        <SectionCard title="Sección A: Análisis Exploratorio y Visualización" subtitle="Nivel Básico-Intermedio">
            <h4 className="text-xl font-bold text-slate-800 mt-4 mb-2">Paso 1: Cargar las Librerías Esenciales</h4>
            <p className="mb-4">Todo proyecto comienza importando nuestras herramientas de trabajo.</p>
            <CodeBlock code={pythonCode.loadLibraries} />
            
            <h4 className="text-xl font-bold text-slate-800 mt-8 mb-2">Paso 2: Introducir la Base de Datos</h4>
            <p className="mb-4">Usaremos el dataset "California Housing". Scikit-learn nos facilita la vida permitiéndonos cargarlo directamente.</p>
            <CodeBlock code={pythonCode.loadDataset} />

            <h4 className="text-xl font-bold text-slate-800 mt-8 mb-2">Paso 3: Limpieza y Exploración Inicial</h4>
            <p className="mb-4">Vamos a entender la estructura y calidad de nuestros datos.</p>
            <CodeBlock code={pythonCode.exploreData} />
            <AnalysisBlock>
              ¡Buenas noticias! No hay valores nulos. Vemos estadísticas como la media, la desviación estándar, los mínimos y máximos de cada característica. Esto ya nos da pistas: por ejemplo, la edad media de las casas (HouseAge) es de unos 28 años.
            </AnalysisBlock>

            <h4 className="text-xl font-bold text-slate-800 mt-8 mb-2">Paso 4: Visualización para Encontrar Patrones</h4>
            <p className="mb-4">Una imagen vale más que mil tablas. Vamos a visualizar nuestros datos.</p>
            <CodeBlock code={pythonCode.histogramPlot} />
            <div className="my-6 p-4 border rounded-lg bg-white shadow-sm">
              <HousingHistogram />
            </div>
            <AnalysisBlock>
              Los histogramas nos muestran que algunas variables como MedInc (ingreso medio) y AveRooms (promedio de habitaciones) están sesgadas a la derecha. La edad de la vivienda (HouseAge) y el precio (MedHouseVal) parecen tener un tope (valores "topeados").
            </AnalysisBlock>

            <CodeBlock code={pythonCode.correlationPlot} />
             <div className="my-6 p-4 border rounded-lg bg-white shadow-sm">
                <CorrelationHeatmap />
            </div>
            <AnalysisBlock>
              El mapa de calor es clave. Observa la fila MedHouseVal. El color rojo intenso en la casilla MedInc (0.69) indica una fuerte correlación positiva: a mayores ingresos, mayor es el precio de la casa. Esto es lógico y nos dice que MedInc será una variable muy importante para nuestro modelo.
            </AnalysisBlock>

            <CodeBlock code={pythonCode.scatterPlot} />
            <div className="my-6 p-4 border rounded-lg bg-white shadow-sm">
                <IncomeScatterPlot />
            </div>
             <AnalysisBlock>
              Este gráfico confirma visualmente la fuerte correlación. Los puntos de color más cálido (rojo) representan precios más altos y tienden a agruparse en la zona de ingresos medios más altos. El parámetro alpha nos ayuda a ver las zonas de mayor densidad de datos.
            </AnalysisBlock>
        </SectionCard>

        <SectionCard title="Sección B: Construyendo un Modelo Predictivo" subtitle="Nivel Avanzado">
          <p className="mb-4">Ahora que entendemos nuestros datos, vamos a la parte emocionante: construir un modelo de Machine Learning que aprenda de estos datos para predecir precios.</p>
          
          <h4 className="text-xl font-bold text-slate-800 mt-4 mb-2">Paso 5: Preparación de Datos para el Modelo</h4>
          <p className="mb-4">Separamos nuestras "pistas" (características, X) del "resultado" que queremos predecir (el precio, y). Luego, dividimos los datos en un conjunto de entrenamiento (para que el modelo aprenda) y uno de prueba (para evaluarlo con datos que no ha visto).</p>
          <CodeBlock code={pythonCode.prepareData} />

          <h4 className="text-xl font-bold text-slate-800 mt-8 mb-2">Paso 6: Entrenamiento de un Modelo de Random Forest</h4>
          <p className="mb-4">Usaremos un modelo llamado Random Forest (Bosque Aleatorio). Es un modelo potente y versátil que funciona bien en muchos problemas. Esencialmente, construye muchos "árboles de decisión" y promedia sus resultados para una predicción más robusta.</p>
          <CodeBlock code={pythonCode.trainModel} />

          <h4 className="text-xl font-bold text-slate-800 mt-8 mb-2">Paso 7: Evaluación del Modelo</h4>
          <p className="mb-4">¿Qué tan bueno es nuestro modelo? Lo probamos con el conjunto de prueba y medimos su error. Usaremos la métrica RMSE (Error Cuadrático Medio Raíz), que nos dice, en promedio, cuánto se equivoca nuestro modelo en las mismas unidades que el precio.</p>
          <CodeBlock code={pythonCode.evaluateModel} />
          <AnalysisBlock>
            Un RMSE de, por ejemplo, 0.49 significa que las predicciones de nuestro modelo se desvían, en promedio, unos $49,000 del precio real de la vivienda. ¡Nada mal para un primer modelo!
          </AnalysisBlock>
        </SectionCard>

        <SectionCard title="(BONUS) Paso 8: Clusterización para Segmentar Vecindarios" subtitle="K-Means">
          <p className="mb-4">Además de predecir, podemos usar técnicas no supervisadas como la clusterización para encontrar grupos naturales en los datos. Por ejemplo, ¿podemos identificar "tipos" de vecindarios basándonos solo en su ubicación geográfica (Latitude y Longitude)?</p>
          <CodeBlock code={pythonCode.kmeansCluster} />
          <div className="my-6 p-4 border rounded-lg bg-white shadow-sm">
                <ClusterMap />
          </div>
          <AnalysisBlock>
             ¡Fantástico! El algoritmo K-Means ha agrupado automáticamente los puntos de datos en 6 regiones geográficas distintas (como Los Ángeles, el Área de la Bahía, etc.) sin saber nada de geografía, solo usando las coordenadas. Esto podría usarse para analizar si el precio de la vivienda varía significativamente entre estos clusters.
          </AnalysisBlock>
        </SectionCard>

        <header className="text-center mt-24 mb-16">
          <h2 className="text-4xl md:text-5xl font-extrabold text-slate-900 mb-4 tracking-tight">
            <span className="block">Segundo Taller Práctico:</span>
            <span className="block text-red-700">Clasificando la Calidad del Vino</span>
          </h2>
          <p className="max-w-3xl mx-auto text-lg md:text-xl text-slate-600">
            Ahora, aplicaremos nuestras habilidades a un problema de clasificación. ¿Podemos construir un modelo que distinga un vino "bueno" de uno "malo" basándose en sus propiedades físico-químicas?
          </p>
        </header>

        <SectionCard title="Sección A: Explorando el Dataset de Vinos" subtitle="Análisis de Características">
            <h4 className="text-xl font-bold text-slate-800 mt-4 mb-2">Paso 1: Cargar el Dataset de Vinos</h4>
            <p className="mb-4">Nuestro viaje enológico comienza cargando el famoso dataset de calidad de vino tinto. Lo importaremos directamente desde su fuente original usando Pandas.</p>
            <CodeBlock code={pythonCode.wine_loadDataset} />
            
            <h4 className="text-xl font-bold text-slate-800 mt-8 mb-2">Paso 2: Exploración y Estadísticas Iniciales</h4>
            <p className="mb-4">Antes de catar, inspeccionamos. Usamos <code>.info()</code> y <code>.describe()</code> para entender las variables (como acidez, azúcar, alcohol) y verificar que nuestros datos estén completos.</p>
            <CodeBlock code={pythonCode.wine_exploreData} />
            <AnalysisBlock>
              El dataset está completo, ¡sin valores nulos! Las estadísticas descriptivas nos muestran el rango de cada característica. Por ejemplo, el contenido de alcohol varía entre 8.4% y 14.9%.
            </AnalysisBlock>

            <h4 className="text-xl font-bold text-slate-800 mt-8 mb-2">Paso 3: Visualizar la Distribución de la Calidad</h4>
            <p className="mb-4">¿Cómo se distribuyen las puntuaciones de calidad? Un histograma o gráfico de conteo es perfecto para esto.</p>
            <CodeBlock code={pythonCode.wine_qualityHistogram} />
            <div className="my-6 p-4 border rounded-lg bg-white shadow-sm">
              <WineQualityHistogram />
            </div>
            <AnalysisBlock>
              La mayoría de los vinos se agrupan en las puntuaciones 5 y 6. Los vinos de alta calidad (7 y 8) y baja calidad (3 y 4) son mucho menos comunes. Esta distribución desigual (desbalanceada) es un desafío importante en muchos problemas de machine learning.
            </AnalysisBlock>

            <h4 className="text-xl font-bold text-slate-800 mt-8 mb-2">Paso 4: Descubrir Correlaciones entre Atributos</h4>
            <p className="mb-4">¿Qué características están más relacionadas con la calidad? Una matriz de correlación nos dará la respuesta.</p>
            <CodeBlock code={pythonCode.wine_correlationPlot} />
             <div className="my-6 p-4 border rounded-lg bg-white shadow-sm flex justify-center">
                <WineCorrelationHeatmap />
            </div>
            <AnalysisBlock>
              El mapa de calor revela que el <strong>alcohol</strong> tiene la correlación positiva más alta con la calidad (0.48). ¡Los vinos con más alcohol tienden a tener mejor puntuación! Por otro lado, la <strong>acidez volátil</strong> tiene una correlación negativa fuerte (-0.39), indicando que es un rasgo indeseable.
            </AnalysisBlock>
        </SectionCard>

        <SectionCard title="Sección B: Construyendo un Modelo Clasificador" subtitle="De 'Malo' a 'Bueno'">
          <p className="mb-4">Ahora, construiremos un modelo que aprenda estas relaciones para clasificar un vino como "bueno" o "malo" automáticamente.</p>
          
          <h4 className="text-xl font-bold text-slate-800 mt-4 mb-2">Paso 5: Preparación de Datos para Clasificación</h4>
          <p className="mb-4">El mundo real raramente es blanco o negro, pero para nuestro modelo, simplificaremos. Convertiremos la escala de calidad (3-8) en una etiqueta binaria: vinos con 7 o más puntos son "buenos" (1), el resto son "malos" (0).</p>
          <CodeBlock code={pythonCode.wine_prepareData} />

          <h4 className="text-xl font-bold text-slate-800 mt-8 mb-2">Paso 6: Entrenamiento de un Modelo de Regresión Logística</h4>
          <p className="mb-4">Usaremos la Regresión Logística, un modelo de clasificación fundamental y eficiente. Para asegurar que el modelo funcione correctamente, también escalaremos las características para que todas tengan una importancia similar al principio. Un `pipeline` de Scikit-learn nos ayuda a hacer esto de forma limpia y ordenada.</p>
          <CodeBlock code={pythonCode.wine_trainModel} />

          <h4 className="text-xl font-bold text-slate-800 mt-8 mb-2">Paso 7: Evaluación del Clasificador</h4>
          <p className="mb-4">¿Nuestro sommelier artificial es preciso? Medimos su rendimiento con el conjunto de prueba, que contiene vinos que nunca antes ha visto.</p>
          <CodeBlock code={pythonCode.wine_evaluateModel} />
          <AnalysisBlock>
            Una precisión del 87% suena bien, pero ¡cuidado! Como vimos, hay muchos más vinos "malos" que "buenos". El <strong>reporte de clasificación</strong> es clave aquí. Nos muestra que el modelo es muy bueno prediciendo vinos "malos" (alto `recall` para 'Vino Malo'), pero le cuesta más identificar correctamente los "buenos" (menor `recall` para 'Vino Bueno'). Mejorar la predicción de la clase minoritaria es un desafío común y un siguiente paso para mejorar el modelo.
          </AnalysisBlock>

          <h4 className="text-xl font-bold text-slate-800 mt-8 mb-2">Paso 8: Predicción y Despliegue Interactivo</h4>
          <p className="mb-4">Una vez entrenado, nuestro modelo está listo para trabajar. Primero, veamos cómo usaríamos el <code>pipeline</code> para predecir la calidad de un nuevo vino con características específicas.</p>
          <CodeBlock code={pythonCode.wine_predictNew} />
          <AnalysisBlock>
              Este código crea un nuevo DataFrame con los datos de un vino hipotético, lo pasa por el <code>pipeline</code> (que aplica el escalado y la predicción) y nos devuelve no solo la clasificación ("Buena Calidad" o "Calidad Regular"), sino también la probabilidad o "confianza" que el modelo tiene en su propia predicción.
          </AnalysisBlock>

          <h5 className="text-lg font-bold text-slate-800 mt-8 mb-2">Creando un Clasificador Interactivo en Colab</h5>
          <p className="mb-4">
              Para hacerlo aún más potente, podemos crear una pequeña interfaz directamente en nuestro cuaderno. El siguiente código utiliza <code>ipywidgets</code> para generar controles deslizantes interactivos, permitiéndote "diseñar" un vino y ver la clasificación al instante.
          </p>
          <CodeBlock code={pythonCode.wine_interactive_ui} />
          <AnalysisBlock>
            Este código crea un slider para cada característica del vino. Al hacer clic en el botón "Clasificar Vino", se recogen los valores, se pasan al <code>pipeline</code> y se muestra la predicción. A continuación, puedes ver una recreación de esta misma idea directamente en la página web.
          </AnalysisBlock>

          <div className="mt-8">
            <WineClassifier />
          </div>

        </SectionCard>


        <SectionCard title="Conclusión y Próximos Pasos">
          <p className="mb-4 text-xl font-semibold text-green-700">¡Felicidades! Has completado un ciclo completo de un proyecto de ciencia de datos:</p>
          <ul className="list-decimal list-inside space-y-2 mb-6">
              <li>Entendiste el problema y los datos.</li>
              <li>Limpiaste y exploraste la información.</li>
              <li>Visualizaste patrones clave.</li>
              <li>Construiste un modelo predictivo para estimar precios.</li>
              <li>Evaluaste su rendimiento.</li>
              <li>Aplicaste una técnica avanzada como la clusterización para segmentar.</li>
          </ul>
           <p className="mb-4">Este es solo el comienzo del viaje. Desde aquí, podrías:</p>
          <ul className="list-disc list-inside space-y-2 mb-6">
              <li>Probar otros modelos de machine learning.</li>
              <li>Realizar un "feature engineering" más avanzado para crear nuevas variables.</li>
              <li>Ajustar los hiperparámetros de tu modelo para mejorar su precisión.</li>
          </ul>
          <p className="text-2xl font-bold text-center mt-8 text-indigo-600">¡El mundo de los datos es tuyo para explorar!</p>
        </SectionCard>

      </main>
    </div>
  );
};

export default App;

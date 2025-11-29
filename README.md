# Trabajo-4-final
Este proyecto analiza cómo el crédito otorgado al sector privado en Piura influye en el desempeño de sus exportaciones, combinando técnicas econométricas, modelos de machine learning y análisis causal.

**Trabajo 1: Exploración de Datos (EDA)**

*a. Descarga de variables macroeconómicas del BCRP*

Se descargó la información principal del BCRP usando la API oficial, asegurando un formato homogéneo para todas las series y un rango temporal consistente. 

Se obtuvieron las siguientes variables: crédito total privado en Piura (credito_total_piura), crédito en soles, crédito en dólares, crédito total nacional, exportaciones de Piura (exportaciones_piura), exportaciones nacionales, y el tipo de cambio. 

Cada serie fue convertida a frecuencia mensual con formato datetime para permitir su integración. Estas variables representan la base financiera y comercial del análisis, ya que describen la evolución del crédito y el desempeño exportador del departamento en el tiempo.

*b. Incorporación de tasas de referencia y expectativas económicas*

Desde la API del BCRP se incorporó la tasa de referencia del Banco Central (tasa_bcrp) y el índice de confianza empresarial (confianza_empresarial), mientras que desde FRED se descargó la tasa de fondos federales de EE.UU. (tasa_fed); del INEI, se extrajó PBI_Piura.

Su presencia es importante porque reflejan las condiciones de liquidez, el costo del crédito y el clima de expectativas que enfrentan las empresas de Piura al tomar decisiones de financiamiento y producción.

*c.Creación de variables de rezagos y tasas de crecimiento*

Para expresar la dinámica temporal de la actividad exportadora y del crédito, se generaron rezagos de ambas variables. Estas transformaciones fueron esenciales para trabajar con tasas en lugar de niveles, especialmente en series con alta volatilidad como las exportaciones.

*d. Alineación temporal, limpieza y consolidación final del dataset*

Todas las series se unificaron en un único DataFrame mensual y se reindexaron para garantizar que cada variable estuviera correctamente alineada en el tiempo. Esta estructura consolidada proporciona una base sólida para estudiar la relación entre el crédito y el comportamiento exportador de Piura. 

*e. Gráficos*

Los histogramas muestran diferencias fuertes en volatilidad: las exportaciones son muy inestables, mientras que el crédito crece de manera suave y controlada. 

El boxplot confirma esta asimetría y evidencia que los movimientos extremos provienen del sector exportador. 

El scatterplot revela una relación prácticamente nula entre crecimiento del crédito y exportaciones, reforzando que su conexión mensual es débil. 

El lineplot permite observar tendencias de largo plazo: el crédito crece sostenidamente, pero las exportaciones no siguen ese patrón. 

el heatmap cuantifica estas relaciones y confirma que la correlación crédito-exportaciones es muy baja, mientras que otras variables macroeconómicas presentan vínculos más claros. 

Estos gráficos, en conjunto, describen un escenario donde el crédito aumenta establemente pero no se traduce de forma inmediata ni proporcional en crecimiento exportador.

**Trabajo 2:Modelo Base (Baseline)**

*a. División Temporal y Preparación del Modelado*

Para evaluar correctamente la capacidad predictiva, los datos se dividieron siguiendo un orden temporal: 75% para entrenamiento y 25% final para prueba. Este procedimiento garantiza que el modelo no vea los datos futuros antes de predecirlos y evita la filtración temporal. 

*b.modelos estimados*

Se estimaron tres aproximaciones: un baseline que predice el valor promedio del crecimiento exportador, un modelo OLS simple que relaciona el crecimiento del crédito rezagado con las exportaciones y un modelo OLS complejo que integra variables adicionales como PBI, tasas de referencia y confianza empresarial. 

El baseline sirve como punto mínimo de comparación para determinar si un modelo realmente aporta valor predictivo, mientras que los modelos OLS permiten interpretar la relación entre variables y evaluar si pueden capturar las fluctuaciones mensuales de la economía.

*c. Resultados de Desempeño*

Los resultados muestran un escenario complejo. El baseline obtuvo un MSE de aproximadamente 524.39, mientras que el modelo OLS simple alcanzó un MSE muy similar de 526.87, lo que indica que ambos presentan niveles comparables de error; sin embargo, el simple mantiene interpretabilidad económica al mostrar un coeficiente significativo del crédito rezagado. 

El modelo complejo tuvo un desempeño inferior con un MSE cercano a 568.58 y un R² negativo, señal de sobreajuste y falta de capacidad predictiva. Los modelos, en general, tienden a predecir valores suaves y cercanos al promedio, mientras que las exportaciones reales muestran saltos abruptos difíciles de anticipar.

*d. Gráfica*

El scatter real vs. predicho muestra puntos alejados de la diagonal, confirmando que los modelos no capturan la magnitud de los movimientos mensuales. 

*e. Conclusiones*

Los modelos evaluados muestran una relación estadísticamente positiva entre el crédito rezagado y el crecimiento de las exportaciones (β_simple = 2.2173, p = 0.032; β_complejo = 3.8283, p = 0.034). 

Sin embargo, el desempeño predictivo fuera de muestra es limitado: el modelo simple alcanza MSE_test = 526.87 y el complejo MSE_test = 552.63, ambos con R²_test negativos. La validación cruzada temporal confirma que el modelo simple es más estable (MSE_CV ≈ 636) que el complejo (MSE_CV ≈ 2339). 

Los gráficos real vs predicho muestran alta dispersión, indicando que la volatilidad de las exportaciones limita la capacidad predictiva mensual. 

En general, existe una señal económica razonable, pero los resultados sugieren que factores externos dominan la dinámica exportadora.

**Trabajo 3: odelos Más Complejos**

*a. PCA como diagnóstico*

Se aplicó un Análisis de Componentes Principales (PCA) exclusivamente como herramienta exploratoria para evaluar redundancia y estructura interna entre las variables macroeconómicas. 

Se imputaron medianas solo para esta etapa, se estandarizaron los predictores y se examinaron los porcentajes de varianza explicada por las primeras componentes. 

El PCA permitió detectar colinealidad, validar la presencia de factores comunes y confirmar si las variables macro aportaban información diferenciada antes de avanzar a los modelos predictivos.

*b. Validación cruzada temporal (TimeSeriesSplit)*

El proyecto implementó un esquema especializado de validación cruzada para series temporales usando TimeSeriesSplit con cinco divisiones consecutivas. 

Este método evita mezclar información futura con el pasado y permite evaluar la estabilidad predictiva real del modelo bajo un escenario temporal realista. 

También se generó el primer split train/test que se utilizaría en todo el pipeline, asegurando consistencia durante el proceso de evaluación.

*C.Estandarización*

Los predictores se estandarizaron aplicando StandardScaler, entrenado únicamente con los datos del conjunto de entrenamiento para evitar filtraciones de información (“data leakage”).

Esta estandarización es fundamental para modelos como Ridge y Lasso, que son sensibles a la escala de las variables. 

*d. Modelos lineales penalizados*

Se ajustaron modelos RidgeCV y LassoCV con búsqueda automática de hiperparámetros utilizando el esquema de validación cruzada temporal. Ridge permitió evaluar relaciones lineales con penalización L2 para controlar colinealidad, mientras que Lasso aplicó penalización L1 para seleccionar automáticamente las variables más relevantes. 

*e. Modelos no lineales (Random Forest y XGBoost)*

Se entrenaron Random Forest y XGBoost, dos métodos capaces de capturar interacciones, no linealidades y efectos de cambio de régimen presentes en la dinámica exportadora. 

Estos modelos aprovechan la estructura temporal y la riqueza de predictores del dataset, ofreciendo una aproximación más flexible que los modelos lineales penalizados y potencialmente más adecuada para fenómenos económicos complejos.

*f. Predicciones y métricas de desempeño*

Los modelos no lineales lograron reducciones significativas de error respecto al baseline y a los modelos lineales, destacando XGBoost como el más preciso, lo que sugiere que la relación entre el crédito y el crecimiento de exportaciones presenta estructuras complejas que los modelos flexibles captan mejor.

*g. Importancia de variables*

Se analizó la contribución de cada predictor mediante coeficientes estandarizados para los modelos lineales penalizados y mediante importancias de variable para los modelos basados en árboles. 

Esta sección mostró qué variables macroeconómicas, crediticias y climáticas tienen mayor peso en la explicación del crecimiento exportador y permitió interpretar de forma económica los patrones aprendidos por los modelos.

*h. Conclusión*

El proyecto muestra que la evolución del crédito, las condiciones macroeconómicas, el entorno nacional y ciertos factores climáticos contienen información relevante para explicar la dinámica exportadora de Piura, pero dicha relación no es puramente lineal. 

Los modelos no lineales, especialmente XGBoost (RMSE ≈ 30.48), lograron captar mejor la complejidad del fenómeno que los modelos lineales penalizados y superaron ampliamente al baseline de tipo Random Walk (RMSE ≈ 51.17). 

La importancia de variables reveló que elementos como las exportaciones nacionales, el tipo de cambio, ciertos rezagos del crédito y el PBI regional juegan un rol destacado, mientras que los efectos del Niño Costero aparecen de manera no lineal y dependiente del contexto. 

En conjunto, los resultados indican que la dinámica exportadora regional responde a múltiples factores interactivos, por lo que enfoques más flexibles y estructuralmente adaptados a series de tiempo permiten una interpretación y predicción más precisa del comportamiento exportador de la región.

**TRABAJO 4: MAG y MLP**

*b. Exploración de la variable de tratamiento*

Antes de construir el DAG, se analizó la distribución de la variable de tratamiento y de la variable continua original de crédito. Se graficó un barplot de tratamiento_credito para visualizar cuántas observaciones caen en los grupos de “bajo crédito” (0) y “alto crédito” (1), y se complementó con un histograma de credito_piura_lag1 incluyendo la línea vertical en la mediana utilizada como punto de corte. 

Esta exploración permite verificar que el split entre alto y bajo crédito es razonable, que no genera un desbalance extremo entre grupos y que la construcción del tratamiento binario está basada en la estructura real de los datos.

*c.Construcción y representación del DAG causal*

En esta parte se formalizó la estructura causal mediante un Grafo Acíclico Dirigido (DAG) implementado con networkx. 

El grafo incluye como nodos a las covariables (ICEN_exp, confianza empresarial, tasas de interés y PBI de Piura), el tratamiento (T, derivado de credito_piura_lag1) y el outcome (Y, exportaciones de Piura). 

Se especificaron aristas desde todas las X hacia T y Y (X → T y X → Y), y desde el tratamiento hacia el outcome (T → Y), reflejando la idea de que los factores macro y climáticos influyen tanto en la decisión de otorgar más o menos crédito como en el desempeño exportador, y que el crédito, una vez determinado, impacta directamente sobre las exportaciones. 

La representación gráfica ayuda a identificar explícitamente los backdoor paths y a justificar que, para estimar el efecto causal del crédito, es necesario controlar por las covariables incluidas en el DAG.

*d.Modelado predictivo con Redes Neuronales (MLP) y comparación con otros modelos*

En esta sección se evaluó el desempeño de una red neuronal tipo MLP para predecir el nivel de exportaciones de Piura utilizando como inputs el crédito rezagado, ICEN_exp, confianza empresarial, tasas de interés y PBI regional. 

Primero se entrenó una regresión lineal como baseline, que obtuvo un R² ≈ 0.1432 y un RMSE ≈ 202.46 sobre el conjunto de prueba. Luego se probó un barrido de arquitecturas MLP con activación ReLU y solver lbfgs, donde la mejor configuración fue un MLP con dos capas ocultas de 30 neuronas cada una [(30, 30)], con R² ≈ 0.0961 y RMSE ≈ 207.95. 

Finalmente, se entrenó un Random Forest (n=200), cuyo desempeño fue inferior (R² ≈ –0.3956, RMSE ≈ 258.39). 

Aunque el MLP no logró superar a la regresión lineal en términos de métricas, la comparación entre modelos sugiere la presencia de cierto grado de no linealidad en las relaciones, y deja abierto el camino a futuros ajustes de hiperparámetros y estrategias más específicas para series temporales.

*e. Conclusión*

Combinó dos enfoques complementarios: por un lado, la formalización causal de la relación entre crédito, entorno macroeconómico, condiciones climáticas y exportaciones mediante un DAG explícito (X → T, X → Y y T → Y), y por otro, la evaluación predictiva de modelos lineales, redes neuronales y Random Forest para aproximar el comportamiento de las exportaciones regionales. 

La construcción del dataset causal y del grafo acíclico permitió justificar de manera transparente qué variables se consideran confusoras y por qué deben controlarse al estimar el impacto del crédito sobre las exportaciones. 

En el frente predictivo, la regresión lineal sigue siendo competitiva y obtiene mejores métricas que el MLP y el Random Forest en la configuración probada, pero las redes neuronales aportan evidencia de no linealidades y posibles interacciones entre crédito, PBI, tasas y Niño Costero que un modelo estrictamente lineal no capta del todo. 

En conjunto, el trabajo muestra que, para entender el rol del crédito en la dinámica exportadora de Piura, es importante tanto explicitar las hipótesis causales (vía DAG) como explorar modelos flexibles que permitan capturar relaciones más complejas entre las variables económicas y climáticas.





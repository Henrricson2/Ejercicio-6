# Clasificación Automática de Neumonía en Radiografías de Tórax
## Basada en descriptores tradicionales y modelos de Deep Learning

---

# 1. Introducción

La identificación temprana y precisa de patologías pulmonares mediante radiografías de tórax es un componente fundamental del diagnóstico clínico. La interpretación manual depende fuertemente de la experiencia del especialista y puede verse limitada en instituciones con alta demanda, escasez de radiólogos o variabilidad interobservador.  

Los sistemas de apoyo al diagnóstico basados en visión por computador ofrecen una alternativa para aumentar la precisión y estandarizar interpretaciones.  

Este proyecto aborda la **clasificación binaria de radiografías** (normales vs neumonía), usando:

1. **Descriptores tradicionales** (HOG, LBP, Haralick, Zernike) + modelos clásicos de ML.  
2. **Modelos profundos preentrenados** (ResNet con Transfer Learning).  

Objetivo: comparar ambos enfoques, evaluando desempeño, ventajas, limitaciones y potencial de generalización.

---

# 2. Marco Teórico

## 2.1. Preprocesamiento de Imágenes Médicas

- **Redimensionamiento:** uniformidad de dimensiones.  
- **Escala de grises:** reduce dimensionalidad manteniendo información clínica.  
- **Normalización:** homogeneiza intensidades entre imágenes.  
- **CLAHE:** mejora contraste local evitando amplificación de ruido.  

Estas técnicas permiten consistencia y comparabilidad entre descriptores tradicionales y redes neuronales.

---

## 2.2. Descriptores Tradicionales

- **HOG:** captura gradientes locales, ideal para bordes y estructuras anatómicas.  
- **LBP:** describe microtexturas robustas a cambios de iluminación.  
- **Haralick / GLCM:** atributos de homogeneidad, contraste y correlación.  
- **Momentos de Zernike:** invariantes a rotación, capturan información global de forma.  

---

## 2.3. Métodos de Clasificación

- **SVM**, **Random Forest**, **XGBoost**, **k-NN**, **Regresión Logística**.  
- Todos requieren vectores de características construidos previamente.

---

## 2.4. Deep Learning — Transfer Learning

- **ResNet preentrenada en ImageNet**.  
- Capas convolucionales como extractor de características + capas finales ajustadas a clasificación binaria.  
- *Skip connections* permiten aprender representaciones complejas sin degradación.

---

# 3. Metodología

Pipeline en tres notebooks:

---

## 3.1. Notebook 01 — Exploratory Analysis & Cleaning

- Carga dataset: 5,856 imágenes.  
- Evaluación tamaños, contraste y nitidez.  
- Preprocesamiento: escala de grises, redimensionamiento, normalización y CLAHE.  
- Conclusiones preliminares sobre calidad de imágenes.

**Observaciones Clave:**

- **Contraste y textura:** neumonía → zonas homogéneas/densas; normal → mayor detalle.  
- **Distribución de intensidades:** KDE/CDF muestran diferencias consistentes.  
- **Intensidad promedio y variabilidad:** mayor en neumonía, mayor dispersión entre imágenes afectadas.  

> ![Ejemplo Exploratory Analysis](ruta/de/la/imagen_exploratory.png)

---

## 3.2. Notebook 02 — Feature Extraction

- **26,244 dimensiones de características** extraídas.  
- Captura gradientes, bordes, texturas y geometría pulmonar.

### HOG
- Detecta costillas, columna, contornos pulmonares.  
- Valores normalizados: -7.58 a 10.  

> ![Ejemplo HOG](ruta/de/la/imagen_HOG.png)

### Momentos de Hu y Geometría
- Área: 22,372 px, Perímetro: 4,243  
- Excentricidad: 0.57, Solidez: 0.60, Relación de aspecto: 1.10  

### LBP
- Textura local robusta a iluminación  
- Diferencia tejido sano vs patológico  

> ![Ejemplo LBP](ruta/de/la/imagen_LBP.png)

### GLCM
- 60 dimensiones  
- Contraste: 9.52, Disimilitud: 1.89, Homogeneidad: 0.49  

### Filtros Gabor
- Detectan patrones según frecuencia/orientación  
- Capturan estructuras anatómicas y detalles finos

### Estadísticas básicas
- Media: 0.52, SD: 0.24, Varianza: 0.058  
- Skewness: -0.26, Kurtosis: -0.92, Entropía: -43.78  

**Conclusión:** conjunto de características robusto y diferenciador.

> ![Resumen de extracción de características](ruta/de/la/imagen_resumen.png)

---

## 3.3. Notebook 03 — Classification

- División train/test, estandarización, entrenamiento clasificadores, GridSearchCV.  
- Evaluación: matriz de confusión, precision, recall, F1, ROC/AUC.  
- Comparación modelos tradicionales vs ResNet.  
- Análisis de importancia de características.

---

# 4. Experimentos

## 4.1. Descriptores Tradicionales
- SVM y Random Forest destacan.  
- Sensibilidad moderada, dependiente de preprocesamiento y ruido.

## 4.2. Deep Learning — ResNet
- Mejor F1-score y generalización.  
- Captura patrones complejos sin ingeniería manual.

---

# 5. Análisis y Discusión

**Comparación:**

| Enfoque         | Ventajas                     | Limitaciones                            |
|-----------------|------------------------------|----------------------------------------|
| Tradicional     | Interpretables, menor dataset | Sensible a ruido/contraste, costoso computacional |
| Deep Learning   | Robustez, aprende patrones complejos | Requiere recursos, menor interpretabilidad directa |

**Limitaciones generales:**
- Sin augmentations fuertes  
- Alta variabilidad visual  
- Falta validación externa  
- Sensible a desbalance  

**Posibles mejoras:**
- Fine-tuning completo de ResNet  
- Incremento de dataset  
- Augmentations realistas  
- Ensemble DL + tradicionales  
- Validación cruzada k-fold

---

# 6. Conclusiones y Resultados

## 6.1. Preprocesamiento
- Fundamental para homogeneidad del dataset y calidad de extracción de características.

## 6.2. Descriptores tradicionales
- Funcionan, pero limitados ante variabilidad anatómica y contraste.

## 6.3. Deep Learning
- ResNet Transfer Learning supera métricas clásicas, aprendizaje robusto sin ingeniería manual.

## 6.4. Diferencias de contraste y textura
- Neumonía: zonas densas y homogéneas  
- Normal: más detalle y uniformidad

## 6.5. Distribución de intensidades
- KDE y CDF confirman patrones consistentes entre clases

## 6.6. Feature Extraction
- HOG, LBP, GLCM, Hu, Gabor robustos  
- Estadísticas básicas reflejan buena discriminación

## 6.7. Rendimiento de modelos

**Deep Learning — ResNet Transfer**

| Métrica      | Valor       |
|-------------|------------|
| Accuracy    | 83.65%     |
| Recall      | 98.7%      |
| F1-Score    | 88.3%      |
| AUC-ROC     | 0.945      |

**Modelos tradicionales**

| Modelo       | Accuracy | F1-Score |
|-------------|----------|----------|
| k-NN        | 79%      | 80.7%    |
| RandomForest| 73%      | 76.5%    |
| Otros       | 64-70%   | 69-75%   |

**Recomendaciones:**  
- Usar ResNet Transfer si recursos lo permiten  
- Tradicionales útiles para escenarios de baja complejidad o recursos limitados

---

# 7. Referencias (APA)

- Dalal, N., & Triggs, B. (2005). *Histograms of oriented gradients for human detection*. CVPR.  
- Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). *Multiresolution gray-scale and rotation invariant texture classification with local binary patterns*. IEEE TPAMI.  
- Haralick, R. M. (1973). *Textural features for image classification*. IEEE TSMC.  
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition*. CVPR.  
- Pizer, S. et al. (1987). *Adaptive histogram equalization and its variations*. CVGIP.

---

# 8. Contribución Individual

- Henrry: desarrollo completo de notebooks, análisis de datos, extracción de características y evaluación de modelos.

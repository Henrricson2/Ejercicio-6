# Clasificación Automática de Neumonía en Radiografías de Tórax: Descriptores Tradicionales y Deep Learning

---

# 1. Introducción

La identificación temprana y precisa de patologías pulmonares mediante radiografías de tórax es esencial para un diagnóstico clínico efectivo. La interpretación manual depende fuertemente de la experiencia del radiólogo y puede verse limitada por:

- Alta demanda de estudios.  
- Escasez de especialistas.  
- Variabilidad interobservador.  

Los sistemas de apoyo al diagnóstico basados en visión por computador permiten aumentar la precisión, estandarizar interpretaciones y asistir en decisiones clínicas.

**Objetivo del proyecto:** Clasificar automáticamente radiografías en dos clases: **normales** y **con neumonía**, comparando enfoques basados en:

1. **Descriptores tradicionales** (HOG, LBP, Haralick, Zernike) con modelos clásicos de Machine Learning.  
2. **Deep Learning** mediante **ResNet Transfer Learning**.

---

# 2. Marco Teórico

## 2.1. Preprocesamiento de Imágenes

El preprocesamiento es crítico para garantizar que los modelos funcionen correctamente:

- **Redimensionamiento:** unifica dimensiones para procesamiento vectorial.  
- **Escala de grises:** reduce dimensionalidad sin perder información relevante.  
- **Normalización:** estabiliza intensidades entre imágenes.  
- **CLAHE:** mejora contraste local y evita sobre-amplificación del ruido.  
- **Augmentation** (recomendado para Deep Learning): rotaciones, flips y zoom para robustecer modelos frente a variabilidad de adquisición.

Estas técnicas aseguran consistencia para descriptores tradicionales y CNN, y reducen el riesgo de overfitting.

---

## 2.2. Descriptores Tradicionales

### ✔ HOG — *Histogram of Oriented Gradients*  
Detecta gradientes locales y bordes, útil para estructuras anatómicas.

### ✔ LBP — *Local Binary Patterns*  
Captura microtexturas robustas ante cambios de iluminación, caracterizando patrones pulmonares.

### ✔ Haralick (GLCM)  
Extrae atributos de textura (contraste, homogeneidad, correlación).

### ✔ Momentos de Zernike  
Descriptores invariantes a rotación, útiles para formas globales anómalas.

---

## 2.3. Clasificadores Tradicionales

- **SVM**  
- **Random Forest**  
- **Gradient Boosting**  
- **k-NN**  
- **Regresión Logística**  

**Nota:** Todos requieren vectores de características previamente construidos y ajuste de hiperparámetros (GridSearchCV).

---

## 2.4. Deep Learning — ResNet Transfer Learning

- Red convolucional profunda preentrenada en ImageNet.  
- Capas finales ajustadas para clasificación binaria (normal vs neumonía).  
- **Skip connections** permiten aprendizaje de representaciones profundas sin degradación.  
- Capacidad para detectar patrones complejos de textura y estructura sin ingeniería manual de características.

---

# 3. Metodología

El pipeline se estructuró en tres notebooks principales:

## 3.1. Notebook 01 — Exploratory Analysis & Cleaning

**Actividades:**

1. Carga del dataset (5.856 imágenes).  
2. Evaluación de distribución de tamaños y resolución.  
3. Visualización de muestras para contraste y nitidez.  
4. Preprocesamiento: escala de grises, redimensionamiento, normalización y CLAHE.  
5. Conclusiones preliminares sobre calidad y homogeneidad de imágenes.

**Resultados Clave:**

- Diferencias de contraste y textura entre clases.  
- Distribución de intensidades: neumonía más concentrada en valores medios; normal más dispersa.  
- KDEs y CDFs muestran patrones consistentes entre clases.  
- Intensidad promedio y variabilidad: mayor en neumonía.  

> ![Ejemplo de análisis exploratorio](ruta/de/la/imagen.png)

---

## 3.2. Notebook 02 — Feature Extraction

- Extracción de **HOG, LBP, Haralick y Zernike**.  
- Ajuste fino de parámetros (bloques, radios, ventanas).  
- Almacenamiento en vectores listos para modelado.  
- Total de características: hasta 26.244 dimensiones, reducidas posteriormente con técnicas de selección de características (Feature Selection).

---

## 3.3. Notebook 03 — Classification

**Actividades:**

1. División train/test (80/20).  
2. Estandarización de características.  
3. Entrenamiento y optimización de modelos clásicos.  
4. Entrenamiento de **CNN y ResNet Transfer**.  
5. Evaluación con métricas completas: Accuracy, Precision, Recall, F1-Score, AUC-ROC.  
6. Análisis de **overfitting** mediante CV_F1 vs Test F1.  
7. Comparación de enfoques tradicionales vs Deep Learning.

---

# 4. Resultados

## 4.1. Descriptores Tradicionales

| Modelo | Accuracy | F1-Score | Test AUC | Overfitting |
|--------|---------|----------|----------|------------|
| k-NN | 0.79 | 0.807 | 0.8726 | 0.113 |
| Random Forest | 0.73 | 0.765 | 0.8626 | 0.149 |
| Naive Bayes | 0.70 | 0.750 | 0.7674 | 0.155 |
| SVM | 0.66 | 0.734 | 0.8864 | 0.217 |
| Logistic Regression | 0.64 | 0.723 | 0.8772 | 0.228 |
| Gradient Boosting | 0.59 | 0.692 | 0.704 | 0.141 |

**Observaciones:**  
- k-NN y Random Forest muestran buen equilibrio entre desempeño y generalización.  
- Modelos lineales como SVM y Logistic Regression tienden a sobreajustar.  
- Gradient Boosting tiene desempeño limitado ante la complejidad del dataset.

---

## 4.2. Deep Learning — ResNet Transfer

| Modelo | Accuracy | Recall | F1-Score | AUC-ROC |
|--------|---------|--------|----------|---------|
| ResNet Transfer | 0.8365 | 0.987 | 0.883 | 0.9446 |
| SimpleCNN | 0.697 | 0.979 | 0.802 | 0.8488 |

**Observaciones:**  
- ResNet Transfer logra el **mejor desempeño global**.  
- SimpleCNN tiene muy alto recall pero menor precisión, indicando sobrepredicción de positivos.  
- Deep Learning captura patrones complejos de forma automática.

---

# 5. Comparación de Modelos

## 5.1. Rendimiento General

- **Mejor modelo absoluto:** ResNet Transfer (DL).  
- **Tradicionales confiables:** k-NN y Random Forest.  
- **Modelos con desempeño limitado:** Gradient Boosting, combinaciones de descriptores seleccionados.

## 5.2. Recomendaciones Prácticas

- Para máxima precisión y sensibilidad: **ResNet Transfer**.  
- Para pruebas rápidas o con recursos limitados: **k-NN o Random Forest**.  
- Considerar balanceo de clases y augmentations en Deep Learning.  
- La selección de características puede reducir dimensionalidad, pero no garantiza mejor desempeño.

---

# 6. Discusión

- Preprocesamiento y análisis exploratorio son esenciales para diferenciar clases y evitar ruido.  
- Deep Learning supera a modelos clásicos en casi todas las métricas.  
- Tradicionales aún útiles en escenarios de bajo recurso o para interpretabilidad.  
- Mejoras futuras: ensemble de CNN + descriptores, fine-tuning de ResNet, validación cruzada k-fold y augmentations.

---

# 7. Conclusiones

1. **Preprocesamiento**: esencial para consistencia y reducción de ruido.  
2. **Descriptores tradicionales**: funcionales, pero limitados frente a la complejidad de las imágenes.  
3. **Deep Learning (ResNet Transfer)**: mejor desempeño y generalización.  
4. **Pipeline reproducible y escalable** para nuevos datasets de radiografías.  
5. **Análisis estadístico y métricas** permiten identificar patrones clave y diferenciar clases automáticamente.  
6. **k-NN y Random Forest**: balance de desempeño y robustez en escenarios con recursos limitados.

---

# 8. Referencias (APA)

- Dalal, N., & Triggs, B. (2005). *Histograms of oriented gradients for human detection*. CVPR.  
- Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). *Multiresolution gray-scale and rotation invariant texture classification with local binary patterns*. IEEE TPAMI.  
- Haralick, R. M. (1973). *Textural features for image classification*. IEEE TSMC.  
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition*. CVPR.  
- Pizer, S. et al. (1987). *Adaptive histogram equalization and its variations*. CVGIP.

---

# 9. Contribución Individual

- Preprocesamiento completo de imágenes (CLAHE, grises, redimensionamiento).  
- Extracción de características HOG, LBP, Haralick y Zernike.  
- Entrenamiento, optimización y evaluación de clasificadores clásicos y ResNet Transfer.  
- Análisis exploratorio, interpretación de resultados y visualizaciones.  
- Elaboración del pipeline reproducible y documentación del proyecto.


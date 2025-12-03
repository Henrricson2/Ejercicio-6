# Clasificación Automática de Neumonía en Radiografías de Tórax: Descriptores Tradicionales y Deep Learning

---

# 1. Introducción

La identificación temprana y precisa de patologías pulmonares mediante radiografías de tórax es esencial para un diagnóstico clínico efectivo. La interpretación manual depende fuertemente de la experiencia del radiólogo y puede verse limitada por:

- Alta demanda de estudios.  
- Escasez de especialistas.  
- Variabilidad interobservador.  

Los sistemas de apoyo al diagnóstico basados en visión por computador permiten aumentar la precisión, estandarizar interpretaciones y asistir en decisiones clínicas.

**Objetivo del proyecto:** Clasificar automáticamente radiografías en dos clases: **normales** y **con neumonía**, comparando dos enfoques:

1. **Descriptores tradicionales** (HOG, LBP, Haralick, Zernike) con modelos clásicos de machine learning.  
2. **Deep Learning** mediante **ResNet Transfer Learning**.

---

# 2. Marco Teórico

## 2.1. Preprocesamiento de Imágenes

El preprocesamiento es crítico para garantizar que los modelos funcionen correctamente:

- **Redimensionamiento:** unifica dimensiones para procesamiento vectorial.  
- **Escala de grises:** reduce dimensionalidad sin perder información relevante.  
- **Normalización:** estabiliza intensidades entre imágenes.  
- **CLAHE:** mejora contraste local y evita sobre-amplificación del ruido.

Estas técnicas aseguran consistencia para descriptores tradicionales y CNN.

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
- **XGBoost**  
- **k-NN**  
- **Regresión Logística**  

**Nota:** Todos requieren vectores de características previamente construidos.

---

## 2.4. Deep Learning — ResNet Transfer Learning

- ResNet preentrenada en ImageNet, con capas convolucionales como extractor de características.  
- Se ajustaron capas finales para clasificación binaria.  
- **Skip connections** permiten aprendizaje de representaciones profundas complejas.

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

**Análisis exploratorio:**

- **Diferencias de contraste y textura:**  
  - Neumonía: zonas homogéneas y densas.  
  - Normal: detalles pulmonares visibles.  
  - Evidencia: CLAHE, Sobel/Laplaciano, colormaps.

- **Distribución de intensidades:**  
  - Neumonía: concentración de píxeles en rangos intermedios (áreas opacas).  
  - Normal: distribución más dispersa (aire y tejido sano).  
  - CDF confirma mayor probabilidad acumulada en valores bajos/medios en neumonía.

- **Comparación agregada:**  
  - KDEs y CDFs muestran diferencias consistentes entre clases.  

- **Intensidad promedio y variabilidad:**  
  - Media ligeramente mayor en neumonía.  
  - Mayor variabilidad entre imágenes afectadas.

- **Importancia:**  
  - Permite identificar patrones visuales y estadísticos antes de entrenar modelos.

> ![Ejemplo de análisis exploratorio](ruta/de/la/imagen.png)

---

## 3.2. Notebook 02 — Feature Extraction

**Descriptores implementados:** HOG, LBP, Haralick, Zernike.  
**Consideraciones técnicas:**  

- Ajuste de parámetros de ventanas, radios y bloques.  
- Vectorización y almacenamiento de características listas para modelado.

---

## 3.3. Notebook 03 — Classification

**Actividades:**  

1. División train/test.  
2. Estandarización de características.  
3. Entrenamiento de modelos clásicos.  
4. Optimización con GridSearchCV.  
5. Evaluación: matrices de confusión, métricas F1, ROC/AUC.  
6. Ranking de modelos tradicionales.  
7. Implementación de CNN (ResNet).  
8. Comparación de enfoques.  
9. Interpretación de importancia de características.

---

# 4. Resultados

## 4.1. Descriptores Tradicionales

- SVM y Random Forest destacan.  
- Limitaciones: sensibilidad moderada, dependencia del descriptor, susceptibilidad a ruido.

## 4.2. Deep Learning — ResNet

- Mayor F1-score y capacidad de generalización.  
- Captura patrones texturales y estructurales complejos.  
- No requiere ingeniería manual.

---

# 5. Comparación de Modelos

## 5.1. Dominio de Deep Learning

- **ResNet Transfer (DL)**:  
  - Accuracy: 83.65%  
  - Recall: 98.7%  
  - F1-Score: 88.3%  
  - AUC-ROC: 0.945  
- Captura patrones complejos, equilibrio entre precisión y sensibilidad.

## 5.2. Modelos tradicionales

- **k-NN:** Accuracy 79%, F1 80.7%  
- **Random Forest:** Accuracy 73%, F1 76.5%  
- Recall alto, Precision más baja.  
- Naive Bayes, SVM, Logistic Regression: Accuracy 64-70%

## 5.3. Modelos con desempeño limitado

- **Gradient Boosting:** Accuracy 59%, F1 69%  
- Difícil captura de la complejidad de los datos.

## 5.4. Recomendaciones

- Deep Learning: máximo rendimiento.  
- Tradicionales: útiles en pruebas rápidas o recursos limitados.  
- Recomendación: **ResNet Transfer** si los recursos lo permiten.

---

# 6. Discusión

- Los descriptores tradicionales funcionan pero dependen de buen preprocesamiento.  
- Deep Learning supera a los modelos clásicos en todas las métricas.  
- El análisis exploratorio es clave para diseñar un pipeline robusto.  
- Posibles mejoras: augmentations, fine-tuning, ensemble CNN + descriptores, validación cruzada k-fold.

---

# 7. Conclusiones

1. Preprocesamiento es esencial.  
2. Descriptores tradicionales permiten clasificadores funcionales, aunque limitados.  
3. ResNet Transfer Learning ofrece mejor desempeño y generalización.  
4. Pipeline reproducible y escalable.  
5. Análisis exploratorio y métricas estadísticamente consistentes ayudan a diferenciar clases automáticamente.

---

# 8. Referencias (APA)

- Dalal, N., & Triggs, B. (2005). *Histograms of oriented gradients for human detection*. CVPR.  
- Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). *Multiresolution gray-scale and rotation invariant texture classification with local binary patterns*. IEEE TPAMI.  
- Haralick, R. M. (1973). *Textural features for image classification*. IEEE TSMC.  
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition*. CVPR.  
- Pizer, S. et al. (1987). *Adaptive histogram equalization and its variations*. CVGIP.

---

# 9. Contribución Individual

- Implementación de preprocesamiento (grises, CLAHE, redimensionamiento).  
- Desarrollo de funciones para HOG, LBP, Haralick y Zernike.  
- Entrenamiento y optimización de clasificadores clásicos y ResNet.  
- Análisis de resultados y visualizaciones.  
- Documentación y elaboración de reportes del pipeline completo.

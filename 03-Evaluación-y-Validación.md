# ⚖️ Módulo 03: Evaluación, Validación y Regularización

En Machine Learning, construir un modelo no solo implica lograr una alta precisión en el conjunto de entrenamiento, sino garantizar su capacidad de **generalización** ante datos no observados. Este módulo explora las bases matemáticas de la capacidad de generalización, las estrategias de validación y las técnicas de regularización para prevenir el sobreajuste.

---

## 🎯 1. Compromiso Sesgo-Varianza (*Bias-Variance Tradeoff*)

El error esperado de un modelo de regresión supervisado en un punto $x$ se puede descomponer matemáticamente en tres componentes fundamentales:

$$
\mathbb{E} \left[ \left( y - \hat{f}(x) \right)^2 \right] = \text{Bias}\left[\hat{f}(x)\right]^2 + \text{Var}\left[\hat{f}(x)\right] + \sigma^2
$$

### Componentes del Error:

1. **Sesgo ($\text{Bias}\left[\hat{f}(x)\right]$):** 
   Diferencia entre la predicción promedio de nuestro modelo y el valor real que intentamos predecir.
   $$\text{Bias}\left[\hat{f}(x)\right] = \mathbb{E}\left[\hat{f}(x)\right] - f(x)$$
   * **Sesgo Alto:** Supuestos simplistas sobre los datos $\rightarrow$ **Subajuste (*Underfitting*)**.

2. **Varianza ($\text{Var}\left[\hat{f}(x)\right]$):** 
   Variabilidad de la predicción del modelo para un punto dado si cambiara el conjunto de entrenamiento.
   $$\text{Var}\left[\hat{f}(x)\right] = \mathbb{E}\left[ \left( \hat{f}(x) - \mathbb{E}\left[\hat{f}(x)\right] \right)^2 \right]$$
   * **Varianza Alta:** Extrema sensibilidad a pequeñas fluctuaciones en el conjunto de entrenamiento $\rightarrow$ **Sobreajuste (*Overfitting*)**.

3. **Error Irreducible ($\sigma^2$):** 
   Ruido inherente al proceso generador de los datos o variables no observadas. No se puede reducir independientemente del algoritmo utilizado.

```text
 Complejidad del Modelo
 Bajo  <---------------------------------------------------------> Alto
 (Subajuste / Underfitting)                         (Sobreajuste / Overfitting)

 Sesgo Alto                                                       Varianza Alta
 Varianza Baja                                                      Sesgo Bajo
 Modelo muy rígido                                         Modelo demasiado flexible
```

---

## 🧪 2. Estrategias de Validación de Modelos

Para estimar correctamente el error de generalización y prevenir el *Data Leakage* (fuga de información), es indispensable dividir apropiadamente los datos.

### 2.1 División Train / Validation / Test
* **Train Set ($\sim 60-80\%$):** Utilizado para optimizar los parámetros del modelo ($\boldsymbol{w}, b$).
* **Validation Set ($\sim 10-20\%$):** Utilizado para ajustar **hiperparámetros** (ej. $\lambda$, profundidad del árbol).
* **Test Set ($\sim 10-20\%$):** Reservado únicamente para la evaluación final imparcial.

### 2.2 Validación Cruzada ($K$-Fold Cross-Validation)
El conjunto de datos de entrenamiento se divide aleatoriamente en $K$ particiones (folds) de igual tamaño:

```text
 Fold 1     Fold 2     Fold 3     Fold 4     Fold 5
+----------+----------+----------+----------+----------+
|   Test   |  Train   |  Train   |  Train   |  Train   |  -> Iteración 1
+----------+----------+----------+----------+----------+
|  Train   |   Test   |  Train   |  Train   |  Train   |  -> Iteración 2
+----------+----------+----------+----------+----------+
 ...
```

Para cada iteración $k \in \{1, \dots, K\}$:
* Se entrena el modelo en $K-1$ bloques.
* Se evalúa en el bloque $k$ restante obtieniendo una métrica $E_k$.

La métrica final de validación es el promediado de las $K$ iteraciones:

$$
E_{\text{CV}} = \frac{1}{K} \sum_{k=1}^{K} E_k
$$

#### Variantes Comunes:
* **Stratified $K$-Fold:** Conserva la proporción de clases en cada fold (crucial en problemas desbalanceados).
* **Leave-One-Out (LOOCV):** Caso extremo donde $K = m$ (útil en conjuntos de datos muy pequeños, aunque costoso computacionalmente).

---

## 🛡️ 3. Regularización ($L_1$ y $L_2$)

La regularización penaliza la complejidad del modelo añadiendo un término de penalización a la función de costo original $J(\boldsymbol{w})$. Esto fuerza a los coeficientes $\boldsymbol{w}$ a tomar valores más pequeños, reduciendo la varianza del modelo.

### 3.1 Regresión Ridge (Regularización $L_2$)
Añade la suma de los cuadrados de los coeficientes a la función de costo:

$$
J_{\text{Ridge}}(\boldsymbol{w}) = \text{MSE}(\boldsymbol{w}) + \alpha \sum_{j=1}^{n} w_j^2 = \text{MSE}(\boldsymbol{w}) + \alpha \|\boldsymbol{w}\|_2^2
$$

* **Comportamiento:**
  * Contrae los coeficientes hacia cero (*shrinkage*), pero **nunca los hace exactamente cero**.
  * Adecuado cuando existen muchas características y la mayoría contribuye al resultado (o hay multicolinealidad alta).

### 3.2 Regresión Lasso (Regularización $L_1$)
Añade la suma de los valores absolutos de los coeficientes a la función de costo:

$$
J_{\text{Lasso}}(\boldsymbol{w}) = \text{MSE}(\boldsymbol{w}) + \alpha \sum_{j=1}^{n} |w_j| = \text{MSE}(\boldsymbol{w}) + \alpha \|\boldsymbol{w}\|_1
$$

* **Comportamiento:**
  * Genera **soluciones esparsas** (*sparsity*): elimina características asignándoles coeficientes exactamente iguales a cero ($w_j = 0$).
  * Funciona como un método automático de **selección de características**.

### 3.3 ElasticNet
Combina las penalizaciones $L_1$ y $L_2$ mediante un hiperparámetro de mezcla $r \in [0, 1]$:

$$
J_{\text{ElasticNet}}(\boldsymbol{w}) = \text{MSE}(\boldsymbol{w}) + r \alpha \|\boldsymbol{w}\|_1 + \frac{1 - r}{2} \alpha \|\boldsymbol{w}\|_2^2
$$

```text
Geometría de las Restricciones:

       Norma L1 (Lasso)                     Norma L2 (Ridge)
             w2                                   w2
              ^                                    ^
              | / \                                |  /---\
              |/   \                               | /     \
       -------+-----+-------> w1            -------+-------+-------> w1
              |\   /                               | \     /
              | \ /                                |  \---/
              |                                    |
  Los contornos de pérdida                     Los contornos de pérdida
  tocan las esquinas en los                    tocan la esfera en puntos
  ejes (w1 o w2 = 0).                          donde w1 y w2 != 0.
```

---

## 📈 4. Diagnóstico con Curvas de Aprendizaje

Las **curvas de aprendizaje** grafican el desempeño del modelo (en función de métricas como el error o el $R^2$) vs. el tamaño del conjunto de entrenamiento.

| Diagnóstico | Curva de Entrenamiento | Curva de Validación | Solución Sugerida |
| :--- | :--- | :--- | :--- |
| **Alto Sesgo (Underfitting)** | Error alto | Error alto, muy cercano al de entrenamiento | Probar modelos más complejos, agregar características polinómicas, reducir regularización. |
| **Alta Varianza (Overfitting)** | Error muy bajo | Error alto, gran brecha (*gap*) respecto al entrenamiento | Conseguir más datos, aplicar regularización ($L_1/L_2$), reducir número de características. |
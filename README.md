# Clase 8: neuronas artificiales para compuertas lógicas

## Objetivo

Entrenar una neurona artificial sencilla para aprender las compuertas AND, OR y NOT, comprobar todas sus combinaciones de entrada y visualizar el error de entrenamiento por época. El cálculo del modelo (suma ponderada, bias, escalón y actualizaciones) usa Python puro; Matplotlib se utiliza únicamente para generar las gráficas.

## Modelo y entrenamiento

La neurona calcula la suma ponderada más el bias:

`neto = sum(w_i * x_i) + b`

La función escalón devuelve `1` cuando `neto >= 0` y `0` en otro caso. Por tanto, en el umbral exacto cero devuelve uno.

Para cada ejemplo, se calcula `error = y_real - y_predicho` y se actualizan los parámetros de esta manera:

`w_nuevo = w_viejo + α * error * x`

`b_nuevo = b_viejo + α * error`

El entrenamiento recorre los ejemplos en orden, actualizando después de cada uno. El error registrado en cada época es la suma de los valores absolutos de los errores observados durante sus actualizaciones, como en el ejemplo de la guía. Se detiene cuando una época completa no requiere ninguna actualización (`error total = 0`); luego el programa verifica explícitamente todas las combinaciones antes de declarar aprendida la compuerta. El máximo es de 100 épocas; si se alcanza sin converger, se informa el fallo.

### Parámetros reproducibles

- Pesos iniciales: `0` para cada entrada.
- Bias inicial: `0`.
- Tasa de aprendizaje: `α = 1`.
- Orden de ejemplos: `00, 01, 10, 11` para AND y OR; `0, 1` para NOT.
- Umbral: cero, con salida `1` si `neto >= 0`.

Estos valores se eligieron para tener una base reproducible común a las tres compuertas. La ejecución y las pruebas confirman que todas convergen con ellos, incluso con `α = 1`. Con el umbral inclusivo, los parámetros en cero producen inicialmente una salida `1`; el entrenamiento corrige las predicciones incorrectas.

## Instalación y ejecución

Se requiere Python 3.10 o posterior. Desde esta carpeta, instala la única dependencia externa:

```powershell
python -m pip install -r requirements.txt
```

Ejecuta el entrenamiento y genera las gráficas:

```powershell
python main.py
```

Ejecuta las pruebas automatizadas:

```powershell
python -m unittest discover -s tests -v
```

Los gráficos de error se guardan en `graficos/error_and.png`, `graficos/error_or.png` y `graficos/error_not.png`.

## Resultados de la ejecución

Verificados ejecutando `python main.py` con los parámetros anteriores:

| Compuerta | Épocas | Error absoluto por época | Resultado |
|---|---:|---|---|
| AND | 6 | `[2, 3, 3, 2, 1, 0]` | Aprendida |
| OR | 4 | `[2, 2, 1, 0]` | Aprendida |
| NOT | 3 | `[1, 1, 0]` | Aprendida |

Las tres tablas de verdad se verifican al final de la ejecución: AND produce `0, 0, 0, 1` para `00, 01, 10, 11`; OR produce `0, 1, 1, 1` para esas mismas entradas; NOT produce `1, 0` para `0, 1`. El programa imprime para cada combinación la salida esperada, la predicha y si coincide. El valor cero al final de cada curva indica que durante la última época no hubo errores ni actualizaciones.

## Preguntas de reflexión de la actividad en clase

1. **¿Qué pasa si se inicializa `w1 = 0`, `w2 = 0`, `b = 0`? ¿La neurona aprende? ¿Por qué?** Sí. Con la convención `z >= 0 → 1`, al inicio predice `1` para todas las entradas. Los errores generan actualizaciones que ajustan pesos y bias; con `α = 1` aprende AND en 6 épocas, OR en 4 y NOT en 3. Pesos cero no impiden el aprendizaje porque el bias también se actualiza.
2. **¿Qué pasa si la tasa de aprendizaje es muy alta, como `1.0`? ¿Aprende o se desestabiliza?** En esta actividad, `α = 1.0` sí converge para AND, OR y NOT con los pesos cero y bias cero. No se desestabiliza en estas pruebas. La respuesta depende de los ejemplos, parámetros y algoritmo; una tasa alta no implica automáticamente divergencia en este perceptrón discreto.
3. **¿Qué pasa si la tasa es muy baja, como `0.001`? ¿Cuántas épocas necesita?** Para medirlo de forma ilustrativa se usaron los pesos sugeridos por la guía (`0.1` por entrada) y bias `0`, límite de 1000 épocas. Con el criterio de una época completa sin errores, AND converge en 34 épocas, OR en 2 y NOT en 102. Una tasa pequeña hace más lentos los cambios; el resultado también depende del umbral y del orden de los datos.
4. **¿Por qué se necesita el bias? ¿Qué pasaría si `b = 0` siempre?** El bias desplaza el umbral y permite que la neurona clasifique correctamente la entrada `(0, 0)` como `0`. Si se fija en cero, la suma ponderada para `(0, 0)` siempre es cero y la escalón devuelve `1`, por lo que AND y OR no pueden representar correctamente sus tablas.
5. **¿Cómo cambiaría el código para aprender OR?** Se mantienen neurona y algoritmo; solo se reemplazan los ejemplos de entrenamiento por `00 → 0`, `01 → 1`, `10 → 1`, `11 → 1`, como en `GATES["OR"]`.

## Preguntas de la actividad independiente

1. **¿Cuántas épocas necesitó cada compuerta para aprender?** Con la inicialización, tasa, orden y criterio documentados, AND necesitó 6 épocas, OR 4 y NOT 3.
2. **¿Fue necesario ajustar la tasa de aprendizaje? ¿Por qué?** No para la configuración principal: `α = 1` permitió que las tres compuertas convergieran dentro de 100 épocas. Con una tasa pequeña se requieren más épocas.
3. **¿Qué diferencias hay entre AND y OR?** AND solo da `1` cuando ambas entradas valen `1`; OR da `1` cuando al menos una vale `1`. Ambas son linealmente separables, pero tienen tablas de verdad y fronteras de decisión distintas. Con estos valores OR necesitó menos épocas que AND.
4. **¿Por qué XOR no puede ser aprendida por una sola neurona?** Una neurona con función escalón solo puede separar las entradas mediante una frontera lineal. XOR requiere clasificar `00` y `11` en una clase, y `01` y `10` en la otra; esas clases no se pueden separar con una sola recta. La prueba automatizada comprueba que XOR no se declara convergido.

La guía sugiere inicialmente `w1 = 0.5`, `w2 = 0.5`, `b = -0.7`, `α = 0.1` para AND; esos parámetros ya resuelven la tabla desde el inicio, por lo que el ejemplo termina en una época sin mostrar descenso del error. También propone probar pesos `0.1` y bias `0` para observar la evolución. Este proyecto usa ceros y `α = 1` en su ejecución principal para permitir comparar consistentemente las tres compuertas; los escenarios alternativos de las respuestas se midieron con el mismo algoritmo.

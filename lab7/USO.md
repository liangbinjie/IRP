# USO - Como correr el programa

## Extraccion de celdas

Para extraer las celdas se utilizara el archivo de `main.py`

Esto requiere que exista imagenes en la carpeta `img/input` y una carpeta en `img/output` vacia.

Esto extrae las celdas de las imagenes y luego generara las celdas en `img/output/{etiqueta}`

```bash
python main.py
```

## Generacion de Momentos Hu y sus tablas

Para generar las tablas de Momentos de Hu, se tiene el programa de `tabla.py`, donde utilizando el programa de `hu.py` que tenemos, va a generar los 7 momentos de hu que tiene una imagen.

Estas imagenes estan almacenadas en `img/hu_input`, entonces una vez extraida las celdas, podemos agarrar cualquier celda y estudiar sus momentos de Hu

Se tiene por defecto que se va a trabajar con la imagen original, luego con una imagen rotada 90 grados, con una imagen reducida a 50% y con una imagen desplazada utilizando mapeo lineal.


```bash
python tabla.py
```
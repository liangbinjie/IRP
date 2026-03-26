#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Laboratorio 1 - Procesar Imagen por Medio de sus Píxeles],
  abstract: [
    This document gives a brief introduction on how to manipulate the pixels of a given image in order to achieve a certain effect. 
    The main goal is to understand how the pixels of an image are represented and how they can be manipulated to achieve a desired outcome.
  ],
  authors: (
    (
      name: "Binjie Liang",
      department: [Introducción al Reconocimiento de Patrones],
      organization: [Instituto Técnologico de Costa Rica],
      location: [San José, Costa Rica],
      email: "bliang@estudiantec.cr"
    ),
  ),
  index-terms: ("opencv", "manipulación de píxeles", "procesamiento de imágenes"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

= Introducción
En este documento se va a detallar el procesamiento de imágenes por medio de la manipulación de píxeles, utilizando la biblioteca de OpenCV para Python.

Se buscar comenzar pintando el primer píxel de color negro y que este vaya pintandolo como un tablero de ajedrez, los espacios que no se pintan van a quedar intactos. Después realizamos lo mismo con el color blanco.

= Implementación
Este laboratorio consiste en pintar una imagen como un tablero de ajedrez, usando los píxeles que contiene la imagen. Para esto se va a utilizar la biblioteca de OpenCV para Python. 

Para abrir o leer una imagen OpenCV proporciona una función que realiza esto, el cual es `imread`. Esta función recibe como parámetro la ruta de la imagen. A continuación el código:

```py
import cv2
image = cv2.imread('lab1_imagen.jpg')
```

Luego, para obtener la resolución de la imagen, se usa la función `shape` @opencv. Esta función retorna el `alto, ancho y canales` de una imagen, en ese orden respectivo. A continuación se muestra un ejemplo en código:

```py
height, width, channels = image.shape
```

Un canal es una capa individual de información de color dentro de una imagen digital. En el caso de una imagen en escala de grises, únicamente existe un canal que representa la intensidad o el nivel de brillo de cada píxel. En cambio, una imagen a color normalmente está compuesta por tres canales que corresponden a los componentes rojo, azul y verde (RBG), los cuales combinados permiten representar los distintos colores de la imagen @opencv_channels_medium.

== Algoritmo para pintar tabla de ajedrez

Para acceder a un píxel de una imagen es sencillo, es solo pasarle el nombre de variable que le asignamos a la imagen y la posición de la imagen. Por ejemplo: `imagen[x][y]`.

Sabiendo eso, podemos pintar un píxel sabiendo la ubicación de este. Es decir, le asignamos un color a un píxel deseado.

El algoritmo para pintarlo de forma que quede como un tablero de ajedrez es verificar que tanto el píxel en la posición `x` (ancho) y en la posición `y` (largo) ambos sean pares. De este modo, como las posiciones empiezan en (0,0), claramente ambos estarán en una posición par, y la posición abajo de este va intercalando.

Para asignarle un color es simplemente igualar el píxel a `[r, b, g] = [255, 255, 255]` para el color negro; `[0, 0, 0]` para el color blanco.

```py
for y in range(height):
  for x in range(width):
      if (x) % 2 == (y) % 2:
          image[y, x] = color
```

== Guardar Imagen
Luego de haber procesado la imagen, queda solo guardarlo, para esto usamos la función `imwrite` @opencv, donde necesita dos parámetros, el primero la ruta y nombre de la imagen y como segundo parámetro, la imagen que modificamos, es decir, la imagen que asignamos al inicio. A continuación, un código de ejemplo:

```py
cv2.imwrite("lab1_imagen1.jpg", image)
```

= Experimento

Para experimentar con lo anterior, tenemos en la @capadoccia-img una imagen de Capadoccia de unos globos aerostáticos.

#figure( 
  image("lab1_imagen.jpg", width: 70%),
  caption: [Imagen de Capadoccia como input @capadoccia]
) <capadoccia-img>

Aplicaremos un pintado de negro en forma de ajedrez. Con esto nos da la imagen @capadoccia-img_b

#figure( 
  image("lab1_imagen1.jpg", width: 70%),
  caption: [Imagen de Capadoccia con filtro negro @capadoccia]
) <capadoccia-img_b>

Como vemos, se ve como si se hubiese aplicado un filtro negro a la imagen.

Si hacemos el mismo proceso, pero con color blanco, tenemos el resultado en la @capadoccia-img_w.

#figure( 
  image("lab1_imagen2.jpg", width: 70%),
  caption: [Imagen de Capadoccia con filtro blanco @capadoccia]
) <capadoccia-img_w>

Podremos apreciar que ahora parece que se aplicó un filtro blanco. 

Si nos acercamos lo suficiente a las imagenes procesadas, veremos que en efecto se encuentra un patrón de cuadro de ajedrez, como se puede ver en @pixels.

#figure( 
  image("pixels.png", width: 70%),
  caption: [Píxeles de la imagen de @capadoccia-img]
) <pixels>

= Análisis
Viendo estos resultados, se puede decir que aplicando un patrón efecto visual de que se ha aplicado un filtro negro o blanco hacia la imagen.

Incluso, se puede modificar el algoritmo para que en vez de que manipule los píxeles con un patrón de ajedrez, pueda manipular todos los píxeles y baje los colores a más oscuros o los también puede subir los colores para que se vea más luminoso.

De este laboratorio se puede aprender que esto es una técnica "medieval" para aplicar filtros blancos y negros a una imagen. 
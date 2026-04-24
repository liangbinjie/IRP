# Laboratorio 05

Deducir z = f (w) (mapeo inverso), de acuerdo a la literatura recomendada definir de forma general los casos
de la funcion de mapeo w = f (z) que sı tienen mapeo inverso.

Desarrolle una funcion que reciba como entradas las constantes complejas a, b, c, d y determine si las mismas
genera una funcion de variable compleja cuyo mapeo inverso sı existe.

Desarrolle una funcion que reciba como entradas una imagen y las constantes complejas a, b, c, d y, tomando
la imagen de entrada como el Plano z genere la representacion de dicho plano en el Plano w.

Desarrolle una funcion que reciba como entradas una imagen y las constantes complejas a, b, asuma que
c = 0 ∧ d = 1, genere el mapeo lineal y demuestre que:
- El mapeo genera una magnificacion cuando b = 0 para todos los casos a ̸= 0 ∧ a ∈ R

- El mapeo genera una magnificacion y una rotacion cuando b = 0 para todos los casos donde
a ̸= 0 ∧ a ∈ C ∧ a / ∈ R
- El mapeo genera unicamente un desplazamiento de todo el Plano z cuando b ̸= 0 ∧ a = 1

- Para el caso donde a ̸= 0 ∧ b ̸= 0, que el mapeo genera la combinacion de una magnificacion, una
rotacion y un desplazamiento de la imagen del Plano z en el Plano w.

---

En este lab y tarea nos referimos a una imagen entrante como plano z

Cada pixel significa un numero complejo => `z = x + jy`

Aplicando el mapeo 

w = (az+b)/(cz+d)

Aplicando el mapeo W obtenemos una nueva posicion `w = u + jv`

Movemos el pixel a esa nueva posicion


Se obtiene:

w(cz+d) = az + b

wcz + wd = az + b

wcz - az = b - wd

z(wc - a) = b - wd

z = (b - wd)/(wc - a)


# Tarea 03
2. Utilizando la biblioteca o herramienta seleccionada, desarrolle una aplicacion que reciba como entradas una
imagen y las constantes complejas a, b, c, d y genere el mapeo directo de la imagen en el Plano w, siempre y
cuando el mapeo exista, guarde la imagen resultante con el nombre imagen2. Use los siguientes valores iniciales
para las constantes complejas: a = 2,1 + j2,1; b = 0; c = 0,003; d = 1 + j.

3. Utilizando el mapeo inverso obtenga los valores de los pixeles faltantes en la imagen generada en el punto 2.,
guarde la imagen resultante como imagen3.

4. Utilizando el mapeo inverso e interpolacion de pixeles en colindancia N=4, obtenga los valores de los pixeles
faltantes en la imagen generada en el punto 2., guarde la imagen resultante como imagen4.

5. Utilizando el mapeo inverso e interpolacion de pixeles en colindancia N=8, obtenga los valores de los pixeles
faltantes en la imagen generada en el punto 2., guarde la imagen resultante como imagen5.

6. Utilizando un filtro Gausseano con una mascara de 5x5, realize el proceso de suavizado de la imagen3, guarde
la imagen resultante como imagen6.

7. Utilizando un filtro Gausseano con una mascara de 5x5, realize el proceso de suavizado de la imagen4, guarde
la imagen resultante como imagen7.

8. Utilizando un filtro Gausseano con una mascara de 5x5, realize el proceso de suavizado de la imagen5, guarde
la imagen resultante como imagen8.

9. Realice un estudio comparativo de las imagenes resultantes de los puntos anteriores, e indique como parte de
su analisis cual de las imagagenes resultantes presenta mejor calidad, justifique su respuesta basandose en la
secuencia de metodos utilizados.

10. Debe de generar un reporte en formato de articulo cientifico de acuerdo al formato IEEE Transactions. En
este reporte agregue un resumen con las 4 conclusiones mas importantes que se deriven del estudio comparativo
anterior.
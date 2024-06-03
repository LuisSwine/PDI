#Importamos las librerías necesarias
import cv2
import matplotlib.pyplot as plt

#Importamos los módulos necesarios
import dependencias
import filtros


# Cargar la imagen
image = cv2.imread('./img/fire.jpg')
cv2.imshow('Imagen Original', image)
cv2.waitKey(0)

#Aumentamos la saturación
factor_saturacion = 1.5
imagen_saturada = filtros.aplicar_filtro_saturacion(image, factor_saturacion)
cv2.imshow('Imagen Saturada', imagen_saturada)
cv2.waitKey(0)

#Aumentamos el contraste
imagen_contrastada = filtros.aumentar_contraste(imagen_saturada)
cv2.imshow('Imagen Contrastada', imagen_contrastada)
cv2.waitKey(0)

# Convertir a espacio de color HSV
hsv_image = cv2.cvtColor(imagen_contrastada, cv2.COLOR_BGR2HSV)

# Definir los valores de umbralización para la saturación
umbral_min = 85  # Ajusta este valor según sea necesario
umbral_max = 255  # Ajusta este valor según sea necesario

# Umbralizar la imagen basada en la saturación
imagen_umbralizada = filtros.umbralizar_saturacion(hsv_image, umbral_min, umbral_max)

# Convertir la imagen umbralizada de HSV a BGR para mostrarla
imagen_umbralizada_bgr = cv2.cvtColor(imagen_umbralizada, cv2.COLOR_HSV2BGR)

# Mostrar las imagenes umbralizadas
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Imagen Umbralizada HSV)')
plt.imshow(cv2.cvtColor(imagen_umbralizada, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Imagen Umbralizada RGB')
plt.imshow(cv2.cvtColor(imagen_umbralizada_bgr, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show() 


#Umbralizamos a partir del canal rojo
rojo_umbral_min = 80  # Ajusta este valor según sea necesario
rojo_umbral_max = 255  # Ajusta este valor según sea necesario
rojo_imagen_umbralizada = filtros.umbralizar_rojo(imagen_umbralizada_bgr, rojo_umbral_min, rojo_umbral_max)

# Mostrar las imagenes umbralizadas
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Imagen Umbralizada RGB')
plt.imshow(cv2.cvtColor(imagen_umbralizada_bgr, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Imagen Umbralizada Rojo')
plt.imshow(cv2.cvtColor(rojo_imagen_umbralizada, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show() 

#Conservamos tonos cálidos y aplicamos el procesamiento final
mask = filtros.conservar_tonos_calidos(rojo_imagen_umbralizada)
imagen_final = dependencias.aplicar_efecto(image, mask)

# Mostrar la imagen original y la imagen de salida
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Imagen Original)')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Imagen final')
plt.imshow(cv2.cvtColor(imagen_final, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show() 

cv2.imwrite('./img/imagen_salida.png', imagen_final)


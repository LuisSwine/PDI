import cv2
import numpy as np

def aumentar_contraste(imagen):
    # Convertir la imagen de BGR a YUV
    img_yuv = cv2.cvtColor(imagen, cv2.COLOR_BGR2YUV)

    # Aplicar ecualización del histograma solo en el canal Y (luminosidad)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # Convertir la imagen de vuelta de YUV a BGR
    img_contraste = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_contraste

def mejorar_tonos_naranjas_y_rojos(imagen):
    # Convertir la imagen de BGR a HSV
    img_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Definir los rangos para los tonos naranjas y rojos
    lower_orange_red = np.array([0, 50, 50])
    upper_orange_red = np.array([25, 255, 255])

    # Crear una máscara para los tonos naranjas y rojos
    mask = cv2.inRange(img_hsv, lower_orange_red, upper_orange_red)

    # Aumentar la saturación y el valor en las áreas seleccionadas por la máscara
    img_hsv[:, :, 1] = np.where(mask > 0, img_hsv[:, :, 1] * 1.5, img_hsv[:, :, 1])
    img_hsv[:, :, 2] = np.where(mask > 0, img_hsv[:, :, 2] * 1.2, img_hsv[:, :, 2])

    # Asegurarse de que los valores de saturación y valor estén en el rango correcto [0, 255]
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)
    img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2], 0, 255)

    # Convertir la imagen de vuelta de HSV a BGR
    img_mejorada = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    return img_mejorada

def aplicar_filtro_saturacion(imagen, factor_saturacion):
    """
    Aplica un filtro de saturación a una imagen.

    :param imagen: Imagen de entrada en formato BGR.
    :param factor_saturacion: Factor de saturación (mayor que 1 aumenta la saturación, entre 0 y 1 la disminuye).
    :return: Imagen con la saturación ajustada.
    """
    # Convertir la imagen de BGR a HSV
    img_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Aplicar el factor de saturación al canal de saturación (S)
    img_hsv[:, :, 1] = img_hsv[:, :, 1] * factor_saturacion

    # Asegurarse de que los valores de saturación estén en el rango correcto [0, 255]
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)

    # Convertir la imagen de vuelta de HSV a BGR
    img_saturada = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    return img_saturada

def conservar_tonos_calidos(imagen):
    # Convertir la imagen de BGR a HSV
    img_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Definir los rangos para los tonos rojizos, naranjas, amarillos, rosas, etc.
    # Estos valores pueden ajustarse según sea necesario
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([179, 255, 255])
    lower_orange = np.array([10, 50, 50])
    upper_orange = np.array([25, 255, 255])
    lower_yellow = np.array([25, 50, 50])
    upper_yellow = np.array([35, 255, 255])
    lower_pink = np.array([140, 50, 50])
    upper_pink = np.array([160, 255, 255])

    # Crear máscaras para cada rango de color
    mask_red1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    mask_orange = cv2.inRange(img_hsv, lower_orange, upper_orange)
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_pink = cv2.inRange(img_hsv, lower_pink, upper_pink)

    # Combinar las máscaras
    mask = mask_red1 | mask_red2 | mask_orange | mask_yellow | mask_pink

    return mask

def umbralizar_saturacion(imagen_hsv, umbral_min, umbral_max):
    # Separar los canales HSV
    h, s, v = cv2.split(imagen_hsv)
    
    # Aplicar el umbral en el canal de saturación
    _, umbral_minima = cv2.threshold(s, umbral_min, 255, cv2.THRESH_BINARY)
    _, umbral_maxima = cv2.threshold(s, umbral_max, 255, cv2.THRESH_BINARY_INV)
    
    # Combinar los umbrales para obtener la máscara final
    mascara = cv2.bitwise_and(umbral_minima, umbral_maxima)
    
    # Aplicar la máscara a la imagen original
    imagen_umbralizada = cv2.bitwise_and(imagen_hsv, imagen_hsv, mask=mascara)
    
    return imagen_umbralizada

def umbralizar_rojo(imagen, umbral_min, umbral_max):
    # Separar los canales RGB
    b, g, r = cv2.split(imagen)
    
    # Aplicar el umbral en el canal rojo
    _, umbral_minima = cv2.threshold(r, umbral_min, 255, cv2.THRESH_BINARY)
    _, umbral_maxima = cv2.threshold(r, umbral_max, 255, cv2.THRESH_BINARY_INV)
    
    # Combinar los umbrales para obtener la máscara final
    mascara = cv2.bitwise_and(umbral_minima, umbral_maxima)
    
    # Aplicar la máscara a la imagen original
    imagen_umbralizada = cv2.bitwise_and(imagen, imagen, mask=mascara)
    
    return imagen_umbralizada


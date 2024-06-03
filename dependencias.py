# Función para mostrar los histogramas de los canales de color
import matplotlib.pyplot as plt
import cv2

def mostrar_histogramas(imagen):
    # Separar los canales de color
    canales = ('b', 'g', 'r')
    colores = ('Azul', 'Verde', 'Rojo')

    # Crear una figura para los histogramas
    plt.figure()
    plt.title('Histogramas de los Canales de Color')
    plt.xlabel('Intensidad de Color')
    plt.ylabel('Número de Píxeles')

    for i, color in enumerate(canales):
        # Calcular el histograma para cada canal
        histograma = cv2.calcHist([imagen], [i], None, [256], [0, 256])
        
        # Graficar el histograma
        plt.plot(histograma, color=color)
        plt.xlim([0, 256])

    plt.legend(colores)
    plt.show()
    
def mostrar_histogramas_hsv(imagen_hsv):
    # Separar los canales HSV
    canales = ('h', 's', 'v')
    colores = ('Hue', 'Saturation', 'Value')

    # Crear una figura para los histogramas
    plt.figure()
    plt.title('Histogramas de los Canales de Color HSV')
    plt.xlabel('Intensidad de Color')
    plt.ylabel('Número de Píxeles')

    for i, color in enumerate(canales):
        # Calcular el histograma para cada canal
        histograma = cv2.calcHist([imagen_hsv], [i], None, [256], [0, 256])
        
        # Graficar el histograma
        plt.plot(histograma, label=colores[i])
        plt.xlim([0, 256])

    plt.legend()
    plt.show()
    
def aplicar_efecto(imagen, mascara):
    # Convertir la imagen original a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen_gris = cv2.cvtColor(imagen_gris, cv2.COLOR_GRAY2BGR)

    # Aplicar la máscara para conservar el color en las áreas deseadas
    imagen_coloreada = cv2.bitwise_and(imagen, imagen, mask=mascara)

    # Invertir la máscara para aplicar el gris en las áreas no deseadas
    mascara_invertida = cv2.bitwise_not(mascara)
    imagen_gris_areas = cv2.bitwise_and(imagen_gris, imagen_gris, mask=mascara_invertida)

    # Combinar las dos imágenes
    imagen_final = cv2.add(imagen_coloreada, imagen_gris_areas)

    return imagen_final
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# Intervalo RGB 1: [203 203 199]
# Intervalo HSV 1: [ 30   5 203]
# Intervalo RGB 2: [214 214 202]
# Intervalo HSV 2: [ 30  14 214]


nome_image = 'BloodImage_00339.jpg'
nome_image = 'BloodImage_00343.jpg'
nome_image = 'BloodImage_00351.jpg'
nome_image = 'BloodImage_00364.jpg'
# nome_image = 'BloodImage_00396.jpg'



image_color = io.imread(nome_image)



def RemoveGlobulosBrancos(img, default=True):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # limites inferiores e superiores das células vermelhas
   
    lower_red = np.array([80, 13, 160])
    upper_red = np.array([150, 50, 200])
    


    # Máscara contendo range (no canal hsv) 
    mask = cv2.inRange(hsv,lower_red,upper_red)

    # mask = cv2.erode(mask, np.ones((2,2),np.uint8), 2)
    k = 10

    # remove os ruídos mínimos  que estão na imagem
    kernel = np.ones((k,k),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
   
    result = cv2.bitwise_and(img, img, mask=mask)


    return result

        



def preprocessamento(hemacias):

    # Conversão para cinza
    gray_redCells = cv2.cvtColor(hemacias, cv2.COLOR_BGR2GRAY)
    
    # Equaliza a imagem
    eq_img = cv2.equalizeHist(gray_redCells)

    # Aplica o filtro gaussiano 5x5 pra remover ruidos indesejados
    k_mask_gaussian = 5
    mask_gaussian = (k_mask_gaussian, k_mask_gaussian)
    
    gaussian_img = cv2.GaussianBlur(eq_img, mask_gaussian, 0)

    return gray_redCells, eq_img, gaussian_img


def binariza_imagem_builtin(img_processada):
    '''
    Binariza a imagem  de forma adaptativa com o uso do método obtivemos os melhores resultados para as imagens selecionadas

    ajustamos para 
    '''
    binary_img = cv2.adaptiveThreshold(img_processada, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 6)
  

    # Aplicacao da erosão  para preencher buracos
    k = 3
    i = 4
    kernel = np.ones((k,k),np.uint8)
    erode_img = cv2.erode(binary_img, kernel, i)

     # Usa o fechamento para remover ruidos
    img_fechada = cv2.morphologyEx(erode_img, cv2.MORPH_CLOSE, kernel)

    inverso = 255-img_fechada

    return binary_img, img_fechada, inverso






def identifica_hemacias(img, binary_image):

    celulas = img.copy()

    circles = cv2.HoughCircles(binary_image,cv2.HOUGH_GRADIENT,1,60,
                             param1=50,param2=17,minRadius=25,maxRadius=60)
    
    hemacias_count = 0

    circles = np.uint32(np.around(circles))
    for i in circles[0,:]:
        cv2.circle(celulas,(i[0],i[1]),i[2],(0,255,0),2)
        cv2.circle(celulas,(i[0],i[1]),2,(0,0,255),3)

        hemacias_count +=1

    
    return celulas, hemacias_count
    




def plota_todo_processo(dados):

    img_original = dados['img_original']
    img_hemacias = dados['img_hemacias']
    img_gray = dados['img_gray']
    img_eq = dados['img_eq']
    img_gaussian = dados['img_gaussian']

    img_binary = dados['img_binary']
    img_erode = dados['erode_close']
    inversa = dados['img_inverse']

    contagem = str (dados['hemacias_count'])
    hemacias = dados['hemacias']

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10,8))
    fig.subplots_adjust(hspace= .2, wspace= .2)

    ax[0][0].imshow(img_original)
    ax[0][0].set_title('Original')

    ax[0][1].imshow(img_hemacias)
    ax[0][1].set_title('Somente hemacias')

    ax[0][2].imshow(img_gray, cmap='gray')
    ax[0][2].set_title('Imagem em tons de cinza')

    ax[1][0].imshow(img_eq, cmap='gray')
    ax[1][0].set_title('Imagem equalizada')

    ax[1][1].imshow(img_gaussian, cmap='gray')
    ax[1][1].set_title('Filtro Gaussiano')

    ax[1][2].imshow(img_binary, cmap='gray')
    ax[1][2].set_title('Imagem binaria')

    ax[2][0].imshow(img_erode, cmap='gray')
    ax[2][0].set_title('Erosão seguida de fechamento')

    ax[2][1].imshow(inversa, cmap='gray')
    ax[2][1].set_title('Inverso')

    ax[2][2].imshow(hemacias)
    ax[2][2].set_title('Hemacias identificadas: ' + contagem)
    

    plt.show()


def deteccao_main(img):
    
    img_hemacias = RemoveGlobulosBrancos(img)

    img_gray, img_eq, img_gaussian = preprocessamento(img_hemacias)

    img_binary, img_erode, img_inverse = binariza_imagem_builtin(img_gaussian)

    hemacias, hemacias_count =  identifica_hemacias(img, img_inverse)

    dados =  {'img_hemacias': img_hemacias,
              'img_original': img,
               'img_gray':img_gray,
               'img_eq': img_eq,
               'img_gaussian': img_gaussian,
               'img_binary': img_binary,
               'erode_close': img_erode,
               'img_inverse': img_inverse,
               'hemacias': hemacias,
               'hemacias_count': hemacias_count
               }
    
    plota_todo_processo(dados)





deteccao_main(image_color)


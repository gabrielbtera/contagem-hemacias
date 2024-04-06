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
# nome_image = 'BloodImage_00351.jpg'
nome_image = 'BloodImage_00402.jpg'
# nome_image = 'BloodImage_00396.jpg'



image_color = io.imread(nome_image)


def manual_inRange(hsv, lower, upper):

    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    
    h_mask = np.logical_and(h >= lower[0], h <= upper[0])
    s_mask = np.logical_and(s >= lower[1], s <= upper[1])
    v_mask = np.logical_and(v >= lower[2], v <= upper[2])
    
    return (h_mask & s_mask & v_mask).astype(np.uint8) * 255



def manual_bitwise_and(img1, img2, mask, negacao=False):
    if img1.shape[:2] != img2.shape[:2] or img1.shape[:2] != mask.shape[:2]:
        raise ValueError("I tamaho das imagens e da mascara deve ser iguais.")
    
    h, w, _ = img1.shape

    result_img = np.zeros((h, w, _), dtype=np.uint8)
    
    for i in range(h):
        for j in range(w):
            mask_tratada = ~mask[i, j] if negacao else mask[i, j] 
            result_img[i, j] = img1[i, j] & mask_tratada
    
    return result_img


def meu_adaptiveThreshold(img, max_level, metodo, type_threshold, block_size, C):
    
    tamanho_borda = block_size // 2

    # Adiciona borda à imagem de entrada
    borda_img = np.pad(img, tamanho_borda, mode='constant')

    # Inicializa a imagem de saída
    img_saida = np.zeros_like(img)

    j, i = img.shape

    # Aplica o limiar adaptativo
    for y in range(tamanho_borda, j + tamanho_borda):
        for x in range(tamanho_borda, i + tamanho_borda):
            roi = borda_img[y - tamanho_borda:y + tamanho_borda + 1, x - tamanho_borda:x + tamanho_borda + 1]
            
            # Aplica o threshold da media
            if metodo == 0:
                threshold_value = np.mean(roi) - C
            # aplica o threshold gaussiano
            else:
                threshold_value = np.mean(roi) - C * np.std(roi) / block_size
            
            # aplica o threshold gaussiano
            if type_threshold == 0:
                img_saida[y - tamanho_borda, x - tamanho_borda] = max_level if img[y - tamanho_borda, x - tamanho_borda] > threshold_value else 0
            
            # aplica o threshold da  media
            else:
                img_saida[y - tamanho_borda, x - tamanho_borda] = 0 if img[y - tamanho_borda, x - tamanho_borda] > threshold_value else max_level

    return img_saida

def RemoveGlobulosBrancos(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    

    # limites inferiores e superiores das células vermelhas
    lower_red = np.array([160, 0, 0])
    upper_red = np.array([180, 161, 247])

    # lower_red = np.array([80, 0, 0])
    # upper_red = np.array([150, 255, 255])
    


    # Máscara contendo range (no canal hsv) 
    # mask = cv2.inRange(hsv,lower_red,upper_red)

    mask = manual_inRange(hsv, lower_red, upper_red)

    # remove os ruídos mínimos  que estão na imagem
    kernel = np.ones((12,12),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
   
    # mask = cv2.bitwise_not(mask)
    result = manual_bitwise_and(img, img, mask=mask, negacao=True)
    # result = cv2.bitwise_and(img, img, mask=mask)



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
    # binary_img = cv2.adaptiveThreshold(img_processada, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 14)

    binary_img = meu_adaptiveThreshold(img_processada, 255, 0, 1, 15, 7)

    # Aplicacao da erosão  para preencher buracos
    k = 5
    i = 5
    kernel = np.ones((k,k),np.uint8)
    img_fechada = cv2.erode(binary_img, kernel, i)

     # Usa o fechamento para remover ruidos
    
    # img_fechada = cv2.morphologyEx(erode_img, cv2.MORPH_CLOSE, kernel)
    # k = 2
    # i = 1
    # kernel = np.ones((k,k),np.uint8)
    # img_fechada = cv2.dilate(img_fechada, kernel, i)


    inverso = 255-img_fechada

    return binary_img, img_fechada, inverso






def identifica_hemacias(img, binary_image):

    celulas = img.copy()

    circles = cv2.HoughCircles(binary_image,cv2.HOUGH_GRADIENT,1,68,
                             param1=50,param2=11,minRadius=30,maxRadius=58)
    
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


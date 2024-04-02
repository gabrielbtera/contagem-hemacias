
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# Intervalo RGB 1: [203 203 199]
# Intervalo HSV 1: [ 30   5 203]
# Intervalo RGB 2: [214 214 202]
# Intervalo HSV 2: [ 30  14 214]


nome_image = 'BloodImage_00339.jpg'
# nome_image = 'BloodImage_00343.jpg'
# nome_image = 'BloodImage_00351.jpg'
# nome_image = 'BloodImage_00364.jpg'
# nome_image = 'BloodImage_00396.jpg'



image_color = io.imread(nome_image)

# def RemoveGlobulosBrancos(img, default=True):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#     # limites inferiores e superiores das células vermelhas
#     if default:
#         lower_red = np.array([80, 13, 170])
#         upper_red = np.array([145, 50, 198])
#     else: 
#         lower_red = np.array([80, 13, 170])
#         upper_red = np.array([145, 50, 199])



#     # Máscara contendo range (no canal hsv) 
#     mask = cv2.inRange(hsv,lower_red,upper_red)

#     # remove os ruídos mínimos  que estão na imagem
#     kernel = np.ones((12,12),np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
   
#     result = cv2.bitwise_and(img, img, mask=mask)


#     return result

def RemoveGlobulosBrancos(img, default=True):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # limites inferiores e superiores das células vermelhas
   
    lower_red = np.array([80, 13, 160])
    upper_red = np.array([150, 50, 200])
    


    # Máscara contendo range (no canal hsv) 
    mask = cv2.inRange(hsv,lower_red,upper_red)

    # mask = cv2.erode(mask, np.ones((2,2),np.uint8), 2)
    k = 9

    # remove os ruídos mínimos  que estão na imagem
    kernel = np.ones((k,k),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
   
    result = cv2.bitwise_and(img, img, mask=mask)


    return result

def binarizacao_limiar_simples(img):
    binary=img.copy()
    limiar = img.max()*(95/256)

    binary[binary<=limiar]=0
    binary[binary > 0]=1

    afterMedian = cv2.medianBlur(img, 3)

    bin = afterMedian > 90


    return bin.astype(np.uint8) * 255

def preprocessamento_sala_aula(img):
    white_cells_img = img.copy()
    remove_red_cells = RemoveGlobulosBrancos(img, False)

    gray_img = cv2.cvtColor(remove_red_cells, cv2.COLOR_BGR2GRAY)

    eq_img = cv2.equalizeHist(gray_img)

    blurry_img = cv2.GaussianBlur(eq_img, (5, 5), 0)

   

    
    binary_img = binarizacao_limiar_simples(blurry_img)



    sobel_img = cv2.Sobel(binary_img, cv2.CV_8U, 1, 1, ksize=15, borderType=cv2.BORDER_REPLICATE) 
    
    k = 2
    i = 4
    kernel = np.ones((k,k),np.uint8)
    eroding_img = cv2.erode(sobel_img, kernel, i)


    # Aplicação de Fechamento para remoção de pontos (ruídos)
    closing_img = cv2.morphologyEx(eroding_img, cv2.MORPH_CLOSE, kernel)

    # Usando o inverso da imagem para detecção das células
    inverse_img = 255-closing_img

    circles = cv2.HoughCircles(inverse_img,cv2.HOUGH_GRADIENT,1, 60,
                             param1=50,param2=16,minRadius=30,maxRadius=59)
    
    
    white_cells_count = 0

    circles = np.uint32(np.around(circles))
    for i in circles[0,:]:
        cv2.circle(white_cells_img,(i[0],i[1]),i[2],(0,255,0),2)
        cv2.circle(white_cells_img,(i[0],i[1]),2,(0,0,255),3)

        white_cells_count +=1

    
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10,7))
    fig.subplots_adjust(hspace= 0.5, wspace= 0.5)

    ax[0][0].imshow(img)
    ax[0][0].set_title('Original')

    ax[0][1].imshow(remove_red_cells)
    ax[0][1].set_title('HSV')

    ax[0][2].imshow(gray_img, cmap='gray')
    ax[0][2].set_title('Gray')

    ax[1][0].imshow(eq_img, cmap='gray')
    ax[1][0].set_title('Equalizacao')

    ax[1][1].imshow(binary_img, cmap='gray')
    ax[1][1].set_title('Binary')

    ax[1][2].imshow(sobel_img, cmap='gray')
    ax[1][2].set_title('Sobel')

    ax[2][0].imshow(closing_img, cmap='gray')
    ax[2][0].set_title('Eroding then Closing')

    ax[2][1].imshow(inverse_img, cmap='gray')
    ax[2][1].set_title('Inverse')

    ax[2][2].imshow(white_cells_img)
    ax[2][2].set_title('White Cells: ' + str(white_cells_count))
    

    plt.show()
        

preprocessamento_sala_aula(image_color)
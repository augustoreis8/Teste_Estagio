import cv2
import numpy as np
from matplotlib import pyplot as plt
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

def find_skeleton3(img):
    skeleton = np.zeros(img.shape,np.uint8)
    eroded = np.zeros(img.shape,np.uint8)
    temp = np.zeros(img.shape,np.uint8)

    retval,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    iters = 0
    while(True):

       cv2.erode(thresh, kernel, eroded)
       cv2.dilate(eroded, kernel, temp)
       cv2.subtract(thresh, temp, temp)
       cv2.bitwise_or(skeleton, temp, skeleton)
       thresh, eroded = eroded, thresh # Swap instead of copy

       iters += 1

       if cv2.countNonZero(thresh) == 0:
         return (skeleton,iters)

imagem1 = cv2.imread('Apenas_Verde1.png')
(canal_h, canal_s, canal_v) = cv2.split(imagem1)
retval, threshold = cv2.threshold(canal_v, 130, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite('Preto_e_Brando.jpg', canal_v)

th=255
minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(threshold,1,np.pi/180,th,minLineLength,maxLineGap)
imagem1 = np.ones((3000,4000,3), np.uint8)
imagem1[imagem1==1]=255
for x in range(0, len(lines)):
    for x1,y1,x2,y2 in lines[x]:
        cv2.line(imagem1,(x1,y1),(x2,y2),(0,0,0),15)
cv2.imwrite('Borda.jpg', imagem1)

imagem1 = cv2.cvtColor(imagem1, cv2.COLOR_BGR2GRAY)
esqueleto, iters = find_skeleton3(imagem1)
esqueleto = cv2.dilate(esqueleto, kernel, iterations=4)
cv2.imwrite('Esqueleto.jpg', esqueleto)


imagem2 = [cv2.imread('Preto e Brando.jpg',0),
           cv2.imread('Borda.jpg',0),
           cv2.imread('Esqueleto.jpg',0)]

imagem2[0] = np.float32(imagem2[0])
imagem2[1] = np.float32(imagem2[1])
imagem2[2] = np.float32(imagem2[2])

corners1 = [cv2.cornerHarris(imagem2[0],2,3,0.04),
            cv2.cornerHarris(imagem2[1],2,3,0.04),
            cv2.cornerHarris(imagem2[2],2,3,0.04)]

imagem3 = [cv2.imread('Preto e Brando.jpg'),
           cv2.imread('Borda.jpg'),
           cv2.imread('Esqueleto.jpg')]

corners2 = [cv2.dilate(corners1[0], None, iterations=3),
            cv2.dilate(corners1[1], None, iterations=3),
            cv2.dilate(corners1[2], None, iterations=3)]

imagem3[0][corners2[0] > 0.01*corners2[0].max()] = [0,255,0]
imagem3[1][corners2[1] > 0.01*corners2[1].max()] = [0,255,0]
imagem3[2][corners2[2] > 0.01*corners2[2].max()] = [0,255,0]

plt.subplot(2,1,2)
plt.imshow(imagem3[0],cmap = 'gray')
plt.imshow(imagem3[1],cmap = 'gray')
plt.imshow(imagem3[2],cmap = 'gray')

plt.imshow(imagem3[0])
plt.show()
plt.imshow(imagem3[1])
plt.show()
plt.imshow(imagem3[2])
plt.show()

cv2.imwrite('Pontos_Preto_e_Branco.jpg', imagem3[0])
cv2.imwrite('Pontos_Borda.jpg', imagem3[1])
cv2.imwrite('Pontos_Esqueleto.jpg', imagem3[2])

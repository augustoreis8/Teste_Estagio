import numpy as np
import cv2
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation, save_json

def baixarImg(imagens):
    imagens['planta_img1'] = cv2.cvtColor(imagens['planta_img1'], cv2.COLOR_BGR2RGB)
    imagens['planta_img2'] = cv2.cvtColor(imagens['planta_img2'], cv2.COLOR_BGR2RGB)
    imagens['planta_img3'] = cv2.cvtColor(imagens['planta_img3'], cv2.COLOR_BGR2RGB)
    imagens['planta_img4'] = cv2.cvtColor(imagens['planta_img4'], cv2.COLOR_BGR2RGB)
    imagens['planta_img5'] = cv2.cvtColor(imagens['planta_img5'], cv2.COLOR_BGR2RGB)
    imagens['planta_img6'] = cv2.cvtColor(imagens['planta_img6'], cv2.COLOR_BGR2RGB)

    hsv = [cv2.cvtColor(imagens['planta_img1'], cv2.COLOR_BGR2HSV),
           cv2.cvtColor(imagens['planta_img2'], cv2.COLOR_BGR2HSV),
           cv2.cvtColor(imagens['planta_img3'], cv2.COLOR_BGR2HSV),
           cv2.cvtColor(imagens['planta_img4'], cv2.COLOR_BGR2HSV),
           cv2.cvtColor(imagens['planta_img5'], cv2.COLOR_BGR2HSV),
           cv2.cvtColor(imagens['planta_img6'], cv2.COLOR_BGR2HSV)]

    imagens['planta_img1'] = cv2.cvtColor(imagens['planta_img1'], cv2.COLOR_BGR2RGB)
    imagens['planta_img2'] = cv2.cvtColor(imagens['planta_img2'], cv2.COLOR_BGR2RGB)
    imagens['planta_img3'] = cv2.cvtColor(imagens['planta_img3'], cv2.COLOR_BGR2RGB)
    imagens['planta_img4'] = cv2.cvtColor(imagens['planta_img4'], cv2.COLOR_BGR2RGB)
    imagens['planta_img5'] = cv2.cvtColor(imagens['planta_img5'], cv2.COLOR_BGR2RGB)
    imagens['planta_img6'] = cv2.cvtColor(imagens['planta_img6'], cv2.COLOR_BGR2RGB)

    mascara = [cv2.inRange(hsv[0], (36, 25, 25), (86, 255, 255)),
               cv2.inRange(hsv[1], (36, 25, 25), (86, 255, 255)),
               cv2.inRange(hsv[2], (36, 25, 25), (86, 255, 255)),
               cv2.inRange(hsv[3], (36, 25, 25), (86, 255, 255)),
               cv2.inRange(hsv[4], (36, 25, 25), (86, 255, 255)),
               cv2.inRange(hsv[5], (36, 25, 25), (86, 255, 255))]

    imascara = [mascara[0] > 0,
                mascara[1] > 0,
                mascara[2] > 0,
                mascara[3] > 0,
                mascara[4] > 0,
                mascara[5] > 0]

    verde = [np.zeros_like(imagens['planta_img1'], np.uint8),
             np.zeros_like(imagens['planta_img2'], np.uint8),
             np.zeros_like(imagens['planta_img3'], np.uint8),
             np.zeros_like(imagens['planta_img4'], np.uint8),
             np.zeros_like(imagens['planta_img5'], np.uint8),
             np.zeros_like(imagens['planta_img6'], np.uint8)]

    verde[0][imascara[0]] = imagens['planta_img1'][imascara[0]]
    verde[1][imascara[1]] = imagens['planta_img2'][imascara[1]]
    verde[2][imascara[2]] = imagens['planta_img3'][imascara[2]]
    verde[3][imascara[3]] = imagens['planta_img4'][imascara[3]]
    verde[4][imascara[4]] = imagens['planta_img5'][imascara[4]]
    verde[5][imascara[5]] = imagens['planta_img6'][imascara[5]]

    cv2.imwrite("Apenas_Verde1.png", verde[0])
    cv2.imwrite("Apenas_Verde2.png", verde[1])
    cv2.imwrite("Apenas_Verde3.png", verde[2])
    cv2.imwrite("Apenas_Verde4.png", verde[3])
    cv2.imwrite("Apenas_Verde5.png", verde[4])
    cv2.imwrite("Apenas_Verde6.png", verde[5])


def main():
    imagens = {'planta_img1': cv2.imread('Imgs/T3R5_20221118_17h.jpg'),
               'planta_img2': cv2.imread('Imgs/T3R5_20221119_13h.jpg'),
               'planta_img3': cv2.imread('Imgs/T3R5_20221119_17h.jpg'),
               'planta_img4': cv2.imread('Imgs/T3R5_20221120_13h.jpg'),
               'planta_img5': cv2.imread('Imgs/T3R5_20221120_17h.jpg'),
               'planta_img6': cv2.imread('Imgs/T3R5_20221121_13h.jpg')}

    altura1, largura1, canais_de_cor1 = imagens['planta_img1'].shape
    altura2, largura2, canais_de_cor2 = imagens['planta_img2'].shape
    altura3, largura3, canais_de_cor3 = imagens['planta_img3'].shape
    altura4, largura4, canais_de_cor4 = imagens['planta_img4'].shape
    altura5, largura5, canais_de_cor5 = imagens['planta_img5'].shape
    altura6, largura6, canais_de_cor6 = imagens['planta_img6'].shape

    coco = Coco()
    coco.add_category(CocoCategory(id=0, name='Plantas'))
    coco_image = [CocoImage(file_name="Apenas_Verde1.png", height=altura1, width=largura1),
                  CocoImage(file_name="Apenas_Verde2.png", height=altura2, width=largura2),
                  CocoImage(file_name="Apenas_Verde3.png", height=altura3, width=largura3),
                  CocoImage(file_name="Apenas_Verde4.png", height=altura4, width=largura4),
                  CocoImage(file_name="Apenas_Verde5.png", height=altura5, width=largura5),
                  CocoImage(file_name="Apenas_Verde6.png", height=altura6, width=largura6)]


    coco_image[0].add_annotation(
        CocoAnnotation(
            bbox=[0, 0, altura1, largura1],
            category_id=0,
            category_name='Plantas'
        )
    )
    coco_image[1].add_annotation(
        CocoAnnotation(
            bbox=[0, 0, altura2, largura2],
            category_id=1,
            category_name='Plantas'
        )
    )
    coco_image[2].add_annotation(
        CocoAnnotation(
            bbox=[0, 0, altura3, largura3],
            category_id=2,
            category_name='Plantas'
        )
    )
    coco_image[3].add_annotation(
        CocoAnnotation(
            bbox=[0, 0, altura4, largura4],
            category_id=3,
            category_name='Plantas'
        )
    )
    coco_image[4].add_annotation(
        CocoAnnotation(
            bbox=[0, 0, altura5, largura5],
            category_id=4,
            category_name='Plantas'
        )
    )
    coco_image[5].add_annotation(
        CocoAnnotation(
            bbox=[0, 0, altura6, largura6],
            category_id=5,
            category_name='Plantas'
        )
    )

    coco.add_image(coco_image[0])
    coco.add_image(coco_image[1])
    coco.add_image(coco_image[2])
    coco.add_image(coco_image[3])
    coco.add_image(coco_image[4])
    coco.add_image(coco_image[5])

    coco_json = coco.json
    save_json(coco_json, "Plantas.json")

    baixarImg(imagens)

main()

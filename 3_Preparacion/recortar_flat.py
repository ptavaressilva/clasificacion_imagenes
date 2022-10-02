#!/usr/bin/env python
# coding=utf-8

from PIL import Image
import os
import shutil
from torchvision import transforms as T
from torch import manual_seed
import random
import math

random_state = 42  # La solución de Douglas Adams para todo
# para torchvision
manual_seed(random_state)
random.seed(random_state)


def get_files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file

# crear réplica de un dataset flat con fotos recortadas a 224x224


def main():

    dataset_path = 'datasets'
    dataset_origin_path = 'flat_10k'
    dataset_output_path = 'flat_10k_224'
    photo_size = 224

    # crear carpetas del dataset de salida
    if os.path.isdir(dataset_path + '/' + dataset_output_path):
        shutil.rmtree(dataset_path + '/' + dataset_output_path, ignore_errors=False,
                      onerror=None)  # eliminar estructura de salida si ya existia
        print('Carpeta {} eliminada'.format(
            dataset_path + '/' + dataset_output_path))

    os.mkdir(
        os.path.join(dataset_path, dataset_output_path)
    )

    categorias = ['drink', 'food', 'inside', 'menu', 'outside']

    for categoria in categorias:
        contador = 0

        os.mkdir(
            os.path.join(dataset_path + '/' + dataset_output_path, categoria)
        )
        print('Carpeta {} creada'.format(
            dataset_path + '/' + dataset_output_path, categoria))
        print('Procesando {}'.format(categoria))
        for file in get_files(dataset_path + '/' + dataset_origin_path + '/' + categoria + '/'):
            contador += 1
            if contador == 50:
                contador = 0
                print('\rProcesando imagen {}', end='')

            with Image.open(dataset_path + '/' + dataset_origin_path + '/' + categoria + '/' + file) as img:
                h, w = img.size
                if (h < w):  # imagen estrecha
                    width = int(photo_size)
                    height = math.floor(photo_size * w/h)
                else:  # imagen ancha
                    width = math.floor(photo_size * h/w)
                    height = int(photo_size)
                pipeline = T.Compose([
                    T.Resize((height, width)),
                    T.RandomResizedCrop((224, 224)),
                ])
                imagen_salida = pipeline(img=img)
                imagen_salida.save(
                    dataset_path + '/' + dataset_output_path + '/' + categoria + '/' + file)


if __name__ == "__main__":
    main()

# Clasificador Multiclase de Imágenes de Restauración

![The Valley (logo](img/TheValley.png)

![Bluetab (logo](img/Bluetab.png)

###  Trabajo final de máster en The Valley Business School (2021-11 / 2022-10)

Equipo: [Pedro Tavares](https://github.com/ptavaressilva), [Toni Vila](https://github.com/tvila), Carlos Cejas y Carlos Huguet.

Tutor: [Javier de la Rosa](https://github.com/jdelarosa91)

En este trabajo evaluamos un conjunto de arquitecturas de **Computer Vision** para clasificar fotos obtenidas de redes sociales, permitiendo su uso en webs y apps.

Trás iterar distintas arquitecturas (CNN y ViT) se usó Transfer Learning para tunear un modelo ResNet, alcanzando un accuracy del 97,2% en validación.

[Memória del proyecyo (pdf)](The\ Valley\ -\ TF\ MDS\ Bluetab\ -\ Grupo\ 2\ -\ Memoria\ del\ Proyecto.pdf)

## Estructura del repositório

**[1_Setup_MLOps](1_Setup_MLOps)** --> Script de arranque del contenedor Docker, notebooks para extraer el dataset y configurar los experimentos en MLflow Tracking.

**[2_EDA](2_EDA)** --> Notebooks para análisis exploratória del dataset.

**[3_Preparacion](3_Preparacion)** --> Notebooks y scripts Python para creación de multiples sub datasets, con distintas características y formatos.

**[4_Modelacion](4_Modelacion)** --> Notebooks y scripts para entrenamento de Redes Neuronales.

**[5_Productivizacion](5_Productivizacion)** --> Aplicaciones web para demostración del proceso de etiquetado automático.

**datasets** --> Carpeta para ficheros de datos usados en los notebooks y scripts Python.

**img** --> Imágenes usadas en este README.

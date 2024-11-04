# Descubriendo el Poder de ​ Langchain: Creando aplicaciones inteligentes
Repositorio con el codigo de la charla

# Requisitos previos
* Tener instalado docker y docker-compose
* Tener una cuenta de [HuggingFace](https://huggingface.co/)
* [Opcional] Tener una cuenta en [Tavily AI](https://app.tavily.com)

# Preparar las variables de entorno

Rellena el fichero .env con tu informacion personal

# Uso de GPU

La imagen de Docker está configurada para usar CUDA 12.0.1. Si la versión de CUDA de tu equipo es incompatible con esa version deberás editar el fichero dockerimage/Dockerfile para cambiarlo a una versión de CUDA que sea compatible.
Puedes buscar las images disponibles en la cuenta de [https://hub.docker.com/r/nvidia/cuda/tags](Nvidia de dockerHub).

# Arrancar el sistema

Basta con ejecutar

```
docker-compose up
```

Tendrás disponible un jupyter lab en el puerto que hayas escogido en el fichero .env


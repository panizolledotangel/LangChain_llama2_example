# LangChain_llama2_example
Repositorio con el codigo de la charla "Descubriendo el Poder de ​ Langchain: Transformación de Texto en Datos Estructurados"

# Requisitos previos
* Tener instalado docker y docker-compose
  
# Acceso LLama-2

1. Rellena el formulario de la pagina de [https://ai.meta.com/llama/#download-the-model](Meta). Usa el mismo correo de tu cuenta de [https://huggingface.co/](HuggingFace).
2. Una vez te hayan dado el OK del formulario anterior, entra en HuggingFace y pide acceso al modelo que quieras usar. 
    a. [https://huggingface.co/meta-llama/Llama-2-7b-chat-hf](7B)
    b. [https://huggingface.co/meta-llama/Llama-2-13b-chat-hf](13B)
    c. [https://huggingface.co/meta-llama/Llama-2-70b-chat-hf](70B)
3. Tardan unas horas, te indicaran que te han dado acceso en la seccion del modelo que hayas elegido en la web de HuggingFace.

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


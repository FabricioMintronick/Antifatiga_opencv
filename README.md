PASOS INICIALES:
1.	Crear la carpeta
cd C:\desarrollo_antifatiga\
2.	Crear entorno virtual
python -m venv venv-antifatiga
3.	Activar el entorno
.\venv-antifatiga\Scripts\actívate
4.	INSTALAR DEPENDENCIAS
pip install opencv-python mediapipe numpy argparse onnxruntime
5.	VERIFICAR INSTALACION:
pip list
6.	ACTUALIZACION DE PIP:
python -m pip install --upgrade pip
7.	DESCARGAR EL MODELO MEDIAPIPE FACELANDMARKER DE GOOGLE:
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
8.	MOVER EL MODELO A:
models/face_landmarker.task
ESTRUCTURA DE CARPETAS Y ARCHIVOS:
 <img width="886" height="445" alt="image" src="https://github.com/user-attachments/assets/268040cd-7dc1-4916-ab5e-e7d62135eaed" />

EJECUCIÓN DEL SISTEMA:
1.	cd C:\desarrollo_antifatiga\
2.	.\venv-antifatiga\Scripts\activate
3.	python antifatiga_main.py --backend mediapipe --show



EN OWASYS EJECUTAR LO SIGUIENTE PARA INSTALAR DEPENDENDICAS:
 pip install -r requirements.txt

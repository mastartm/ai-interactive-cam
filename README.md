# AI-INTERACTIVE-CAM
Bu proje, Python ve MediaPipe kullanarak gerçek zamanlı el ve yüz takibi yapan etkileşimli bir kamera uygulamasıdır,proje geliştirilme aşamasındadır.

Programın çalışması için proje klasöründe model dosyalarının olması gerekmektedir.

Link1:https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
Link2:https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

Temiz bir çalışma ortamı için venv kurulumu yapılması önerilir. Terminale sırasıyla aşağıdaki komutları girin:

python -m venv venv

.\venv\Scripts\activate

pip install opencv-python mediapipe numpy

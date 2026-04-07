import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import datetime

# --- AYARLAR VE MODEL YOLLARI (Buralar aynı kalıyor) ---
face_model_path = 'face_landmarker.task'
hand_model_path = 'hand_landmarker.task'

face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=face_model_path),
    running_mode=vision.RunningMode.VIDEO, num_faces=1
)
hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=hand_model_path),
    running_mode=vision.RunningMode.VIDEO, num_hands=2
)

window_name = 'MediaPipe - Etkileşimli Takip'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with vision.FaceLandmarker.create_from_options(face_options) as face_landmarker, \
     vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker:
    sayac=0
    efekt_acik = True
    goz_kilitli = False
    
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

        face_result = face_landmarker.detect_for_video(mp_image, timestamp)
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp)

        # 1. YÜZ ÇİZİMİ
        if face_result.face_landmarks:
            for face_lms in face_result.face_landmarks:
                h, w, _ = frame.shape
                
                def px(n):
                    return int(face_lms[n].x * w), int(face_lms[n].y * h)

                # --- GÖZ KONTROLÜ (Efekt aç/kapat) ---
                ust_kapak = face_lms[159].y
                alt_kapak = face_lms[145].y
                
                # Mesafe kontrolü
                if abs(ust_kapak - alt_kapak) < 0.015:
                    if not goz_kilitli:
                        efekt_acik = not efekt_acik
                        goz_kilitli = True
                else:
                    goz_kilitli = False

                # --- EFEKTLERİ ÇİZ (SADECE efekt_acik TRUE İSE) ---
                if efekt_acik:
                    # A. PALYAÇO BURNU
                    burun = px(1)
                    cv2.circle(frame, burun, 18, (0, 0, 255), -1)
                    cv2.circle(frame, (burun[0]-6, burun[1]-6), 6, (255, 255, 255), -1)

                    # B. SİBER GÖZLER
                    sol_goz = px(468)
                    sag_goz = px(473)
                    cv2.circle(frame, sol_goz, 9, (255, 255, 0), 2)
                    cv2.circle(frame, sol_goz, 3, (255, 255, 255), -1)
                    cv2.circle(frame, sag_goz, 9, (255, 255, 0), 2)
                    cv2.circle(frame, sag_goz, 3, (255, 255, 255), -1)

                    # C. ALIN TAKISI
                    alin = px(10)
                    cv2.circle(frame, alin, 10, (0, 255, 0), -1) 
                    cv2.circle(frame, (alin[0]-3, alin[1]-3), 4, (200, 255, 200), -1)

        # 2. EL ÇİZİMİ VE ETKİLEŞİM HESAPLAMA
        total_fingers = 0
        
        
        if hand_result.hand_landmarks:
            for hand_lms in hand_result.hand_landmarks:
                # 1. Parmakları Say (İşaret, Orta, Yüzük, Serçe)
                # Sadece bu 4 parmağı döngüyle sayıyoruz
                fingers = []
                tips = [8, 12, 16, 20]
                pip_joints = [6, 10, 14, 18]
                
                for i in range(4):
                    if hand_lms[tips[i]].y < hand_lms[pip_joints[i]].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # 2. Başparmak Analizi (ÖZEL)
                thumb_tip = hand_lms[4]
                thumb_ip = hand_lms[3] # Bir alt boğum
                thumb_base = hand_lms[2] # Kök

                # BAŞPARMAK AÇIK MI? (Yana doğru açılma kontrolü - Sağ/Sol el fark edebilir)
                # Genelde x eksenindeki farka bakılır
                if abs(thumb_tip.x - hand_lms[5].x) > 0.05: 
                    thumb_opened = 1
                else:
                    thumb_opened = 0
                
                # TOPLAM SAYI
                current_hand_fingers = sum(fingers) + thumb_opened
                total_fingers += current_hand_fingers
                if total_fingers==5:
                    sayac+=1
                    kalan_sure =5-(sayac//30)
                    if kalan_sure>0:
                        cv2.putText(frame,f"Gulumse!{kalan_sure}",(500,350),cv2.FONT_HERSHEY_DUPLEX,3,(0,255,255),5)
                    if sayac>=150:
                        dosya_adi = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") #Örn:20231027_143005
                        cv2.imwrite(f"selfie_{dosya_adi}.jpg", frame)
                        print("Foto Çekildi!")
                        sayac=0
                else:
                    sayac=0

                # 3. SADECE BAŞPARMAK İLE LIKE/DISLIKE (Kritik Düzeltme)
                # Diğer parmaklar kapalıyken (yumruk) sadece başparmak yukarıdaysa LIKE
                if thumb_tip.y < thumb_ip.y - 0.05 and sum(fingers) == 0:
                    cv2.putText(frame, "GERCEK LIKE!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                elif thumb_tip.y > thumb_ip.y + 0.05 and sum(fingers) == 0:
                    cv2.putText(frame, "GERCEK DISLIKE!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

        # Ekrana toplam parmak sayısını yaz
        cv2.putText(frame, f"Parmak Sayisi: {total_fingers}", (50, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        # 10 parmak açıksa efekt (Ekranın etrafına çerçeve çizelim)
        if total_fingers == 10:
            cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (0, 255, 255), 20)
            cv2.putText(frame, "MUHTESEM!", (450, 350), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 255), 5)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
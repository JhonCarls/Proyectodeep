import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import re
import logging

# --- CONFIGURACIÓN PARA SILENCIAR PADDLEOCR ---
# Como 'show_log=False' ya no funciona, usamos esto para que no llene la consola
logging.getLogger("ppocr").setLevel(logging.ERROR)

# --- CARGAR MODELOS ---
print("Cargando modelos, por favor espera...")

# Inicializar YOLO
model = YOLO("best.pt") 

# Inicializar OCR
# Nota: He quitado 'show_log=False' para arreglar tu error.
# Si te sigue saliendo el warning amarillo sobre 'use_angle_cls', puedes ignorarlo, 
# el programa funcionará igual.
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# --- CONFIGURACIÓN DE VIDEO ---
# Usa 0 para webcam. Si tienes un video, pon la ruta: "video.mp4"
video_path = "Inputs/video1.mp4"
cap = cv2.VideoCapture(video_path)
#cap = cv2.VideoCapture(0)

# Compilar regex una vez para eficiencia
whitelist_pattern = re.compile(r'^[A-Z0-9]+$')

print("Iniciando detección en tiempo real. Presiona 'q' para salir.")

while True:
    # 1. Leer frame de la cámara
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el frame de la cámara.")
        break

    # 2. Ejecutar YOLO (stream=True es mejor para video)
    results = model(frame, stream=True, verbose=False)

    for result in results:
        for box in result.boxes:
            # Obtener clase y confianza
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # Filtrar: Clase 0 (placa) y Confianza > 0.60
            if cls == 0 and conf > 0.60:
                # Coordenadas
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                # --- PROTECCIÓN DE BORDES (Evita errores si la placa toca el borde) ---
                h_img, w_img = frame.shape[:2]
                pad = 10 
                
                # Aseguramos que las coordenadas con padding no se salgan de la imagen
                y1_crop = max(0, y1 - pad)
                y2_crop = min(h_img, y2 + pad)
                x1_crop = max(0, x1 - pad)
                x2_crop = min(w_img, x2 + pad)

                # Recortar la placa
                plate_image = frame[y1_crop:y2_crop, x1_crop:x2_crop]

                # Si el recorte es válido, pasamos al OCR
                if plate_image.size > 0:
                    try:
                        # 3. Ejecutar OCR
                        ocr_result = ocr.ocr(plate_image, cls=False) # cls=False es más rápido para video

                        if ocr_result and ocr_result[0]:
                            # Extraer texto detectado
                            text_list = [line[1][0] for line in ocr_result[0]]
                            full_text = "".join(text_list)
                            
                            # Limpieza: Solo mayúsculas y números
                            clean_text = "".join([c for c in full_text if c.isalnum()]).upper()

                            # --- FORMATO DE SALIDA (BR-GAY...) ---
                            if len(clean_text) >= 3:
                                # Pone el guion después de los primeros 2 caracteres
                                final_text = f"{clean_text[:2]}-{clean_text[2:]}"
                            else:
                                final_text = clean_text

                            # Imprimir en consola
                            print(f"Placa: {final_text}")

                            # --- DIBUJAR EN PANTALLA ---
                            # Cuadro verde
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Fondo negro para el texto
                            (text_w, text_h), _ = cv2.getTextSize(final_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                            cv2.rectangle(frame, (x1, y1 - 30), (x1 + text_w, y1), (0, 255, 0), -1)
                            
                            # Texto
                            cv2.putText(frame, final_text, (x1, y1 - 5), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                            
                            # Mostrar recorte pequeño (opcional)
                            cv2.imshow("Recorte", plate_image)

                    except Exception as e:
                        # Si falla el OCR en un frame, no pasa nada, seguimos al siguiente
                        pass

    # 4. Mostrar el video
    cv2.imshow("Deteccion Placas", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpieza final
cap.release()
cv2.destroyAllWindows()
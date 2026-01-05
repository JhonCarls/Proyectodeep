import cv2
import queue
import threading
import time
import json
import re
import os
import easyocr
import numpy as np
from ultralytics import YOLO

# --- CONFIGURACIÓN ---
VIDEO_PATH = "./Inputs/video1.mp4"
JSON_OUTPUT = "placas_finales.json"
MAX_CHARS = 8
CONF_OCR_THRESHOLD = 0.4 # Confianza mínima del OCR para aceptar el texto

# --- CARGA DE MODELOS ---
print("Cargando YOLO y EasyOCR... Espere...")

# Modelo YOLO (Detección)
model = YOLO("best.pt")

# Modelo EasyOCR (Lectura)
# gpu=True es crucial si tienes tarjeta gráfica NVIDIA. Si no, pon gpu=False (será más lento el OCR, pero el video seguirá fluido por los hilos)
reader = easyocr.Reader(['en'], gpu=True) 

# --- VARIABLES COMPARTIDAS (THREAD SAFE) ---
# Cola para enviar imágenes al worker de OCR. maxsize=2 evita que se acumulen tareas viejas
ocr_queue = queue.Queue(maxsize=2)
# Diccionario donde se guardan los resultados: { track_id: "PLACA123" }
plate_results = {}
# Set para guardar en JSON sin repetir
saved_plates = set()

# Cargar JSON previo si existe
if os.path.exists(JSON_OUTPUT):
    try:
        with open(JSON_OUTPUT, 'r') as f:
            saved_plates = set(json.load(f))
    except:
        pass

# --- FUNCIÓN DE LIMPIEZA DE TEXTO ---
def clean_plate_text(text):
    # Solo mayúsculas y números
    text = text.upper()
    cleaned = re.sub(r'[^A-Z0-9]', '', text)
    # Regla: Máximo 8 caracteres, mínimo 3
    if 3 <= len(cleaned) <= MAX_CHARS:
        return cleaned
    return None

# --- WORKER: PROCESO DE OCR EN SEGUNDO PLANO ---
def ocr_worker():
    """Este hilo se dedica SOLO a leer placas sin congelar el video"""
    while True:
        try:
            # Esperar imagen de la cola
            task = ocr_queue.get(timeout=1)
            track_id, plate_img = task
            
            # Si ya tenemos un resultado definitivo para este ID, saltar (optimización)
            if track_id in plate_results and len(plate_results[track_id]) >= 3:
                ocr_queue.task_done()
                continue

            # Ejecutar EasyOCR
            # allowlist acelera MUCHO el proceso limitando los caracteres a buscar
            result = reader.readtext(plate_img, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', detail=1)

            for (bbox, text, prob) in result:
                if prob > CONF_OCR_THRESHOLD:
                    final_text = clean_plate_text(text)
                    if final_text:
                        # Guardar en memoria viva
                        plate_results[track_id] = final_text
                        
                        # Guardar en JSON si es nueva
                        if final_text not in saved_plates:
                            saved_plates.add(final_text)
                            with open(JSON_OUTPUT, 'w') as f:
                                json.dump(list(saved_plates), f, indent=4)
                            print(f"--> [NUEVA DETECTADA]: {final_text}")
            
            ocr_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error en OCR thread: {e}")

# Iniciar el hilo del OCR
t = threading.Thread(target=ocr_worker, daemon=True)
t.start()

# --- PROCESO PRINCIPAL (VIDEO) ---
cap = cv2.VideoCapture(VIDEO_PATH)
cv2.namedWindow("ANPR Ultra Fluido", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ANPR Ultra Fluido", 1280, 720)

print("Iniciando video...")

while True:
    success, frame = cap.read()
    if not success:
        break

    # 1. Detección y Rastreo (YOLO)
    # persist=True mantiene el ID del auto
    results = model.track(frame, persist=True, verbose=False, classes=[0], conf=0.5)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # Verificar si ya sabemos qué placa es
            text_to_show = f"ID:{track_id} ..."
            color = (0, 165, 255) # Naranja (procesando)

            if track_id in plate_results:
                text_to_show = plate_results[track_id]
                color = (0, 255, 0) # Verde (leída)
            else:
                # Si NO sabemos la placa, enviamos la imagen al hilo de OCR
                # Usamos put_nowait para que si la cola está llena, NO BLOQUEE el video
                try:
                    # Recorte con margen
                    h, w, _ = frame.shape
                    crop = frame[max(0, y1-10):min(h, y2+10), max(0, x1-10):min(w, x2+10)]
                    ocr_queue.put_nowait((track_id, crop))
                except queue.Full:
                    pass # El OCR está ocupado, no pasa nada, intentamos en el siguiente frame

            # Dibujar (Visualización)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text_to_show, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Mostrar imagen
    cv2.imshow("ANPR Ultra Fluido", frame)
    
    # Control de salida
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
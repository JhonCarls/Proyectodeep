import cv2
import time
import json
import numpy as np
from ultralytics import YOLO

# --- CONFIGURACIÓN ---
VIDEO_PATH = "./Inputs/video2.mp4"
JSON_OUTPUT = "placas_validadas.json"
MAX_CHARS = 8

# --- ¡IMPORTANTE! AJUSTA LOS IDs DE TUS CLASES AQUÍ ---
# Revisa tu data.yaml para saber qué número es cada cosa
CLASS_ID_PLATE = 0  # Ejemplo: 0 es la placa
CLASS_ID_CAR = 1    # Ejemplo: 1 es el vehículo
# Si tu modelo solo detecta placas, necesitarás cargar un yolov8n.pt extra para los autos.
# Asumo que tu 'best.pt' detecta AMBOS.

# --- CARGAR MODELOS ---
print("Cargando modelos...")
# Modelo 1: Detecta Autos y Placas (Tu modelo entrenado)
main_model = YOLO("best.pt")

# Modelo 2: Detecta Caracteres (El modelo Nano que entrenaste)
char_model = YOLO("best_char.pt") 

# --- MEMORIA ---
track_history = {} 
saved_plates = set()

# Cargar JSON previo
try:
    with open(JSON_OUTPUT, 'r') as f:
        saved_plates = set(json.load(f))
except:
    pass

def is_plate_inside_car(plate_box, car_box):
    """
    Verifica si el centro de la placa está dentro de la caja del auto.
    plate_box: [x1, y1, x2, y2]
    car_box: [x1, y1, x2, y2]
    """
    px1, py1, px2, py2 = plate_box
    cx1, cy1, cx2, cy2 = car_box
    
    # Calcular centro de la placa
    plate_center_x = (px1 + px2) / 2
    plate_center_y = (py1 + py2) / 2
    
    # Verificar si el centro está dentro de las coordenadas del auto
    if (cx1 < plate_center_x < cx2) and (cy1 < plate_center_y < cy2):
        return True
    return False

# --- PROCESAMIENTO ---
cap = cv2.VideoCapture(VIDEO_PATH)
cv2.namedWindow("ANPR Lógica Avanzada", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ANPR Lógica Avanzada", 1280, 720)

print("Iniciando video con validación de vehículo...")

while True:
    start_time = time.time()
    success, frame = cap.read()
    if not success:
        break

    # 1. DETECCIÓN GENERAL (Autos + Placas)
    # Detectamos ambas clases. Confianza 0.5 para filtrar basura.
    results = main_model.track(frame, persist=True, verbose=False, conf=0.5)
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Separar detecciones en dos listas
        cars = []
        plates = []

        for box, cls, trk_id in zip(boxes, classes, track_ids):
            if cls == CLASS_ID_CAR:
                cars.append({'box': box, 'id': trk_id})
            elif cls == CLASS_ID_PLATE:
                plates.append({'box': box, 'id': trk_id})

        # 2. DIBUJAR AUTOS (Opcional, para ver qué pasa)
        for car in cars:
            x1, y1, x2, y2 = map(int, car['box'])
            # Dibujamos el auto en gris suave
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

        # 3. PROCESAR PLACAS SOLO SI ESTÁN DENTRO DE UN AUTO
        for plate in plates:
            px1, py1, px2, py2 = map(int, plate['box'])
            p_id = plate['id']
            
            # -- VALIDACIÓN DE CONTENCIÓN --
            valid_plate = False
            parent_car = None
            
            for car in cars:
                if is_plate_inside_car(plate['box'], car['box']):
                    valid_plate = True
                    parent_car = car
                    break # Encontramos el auto dueño de esta placa
            
            if not valid_plate:
                # Si la placa no está dentro de un auto, LA IGNORAMOS (o la pintamos roja de error)
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 1)
                continue # Saltamos al siguiente ciclo, no gastamos recursos leyendo esto

            # --- SI ES VALIDA, PROCEDEMOS ---
            # Dibujar línea visual entre placa y auto (Feedback visual pro)
            cx, cy = int((px1+px2)/2), int((py1+py2)/2)
            car_cx = int((parent_car['box'][0] + parent_car['box'][2])/2)
            car_cy = int(parent_car['box'][1]) # Parte superior del auto
            cv2.line(frame, (cx, cy), (car_cx, car_cy), (0, 255, 255), 1)

            display_text = f"ID:{p_id}"
            box_color = (0, 165, 255) # Naranja (Procesando)

            # A. Revisar Memoria
            if p_id in track_history:
                display_text = track_history[p_id]
                box_color = (0, 255, 0) # Verde (Listo)
            
            # B. Leer Caracteres (Solo si es nueva)
            else:
                h, w, _ = frame.shape
                pad = 4
                crop = frame[max(0, py1-pad):min(h, py2+pad), max(0, px1-pad):min(w, px2+pad)]
                
                # Inferencia modelo de caracteres
                char_results = char_model(crop, verbose=False, conf=0.5)
                
                if len(char_results[0].boxes) > 0:
                    c_boxes = char_results[0].boxes.xyxy.cpu().numpy()
                    c_cls = char_results[0].boxes.cls.int().cpu().tolist()
                    
                    # Ordenar izquierda a derecha
                    sorted_chars = sorted(zip(c_boxes, c_cls), key=lambda c: c[0][0])
                    
                    plate_text = ""
                    for _, cls_id in sorted_chars:
                        if cls_id < len(char_model.names):
                            plate_text += char_model.names[cls_id]
                    
                    if 3 <= len(plate_text) <= MAX_CHARS:
                        track_history[p_id] = plate_text
                        display_text = plate_text
                        box_color = (0, 255, 0)
                        
                        if plate_text not in saved_plates:
                            saved_plates.add(plate_text)
                            print(f"--> [NUEVA PLACA VALIDADA]: {plate_text} (En Auto ID: {parent_car['id']})")
                            with open(JSON_OUTPUT, 'w') as f:
                                json.dump(list(saved_plates), f, indent=4)

            # Dibujar resultado final
            cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)
            cv2.putText(frame, display_text, (px1, py1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    # FPS
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cv2.imshow("ANPR Lógica Avanzada", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import time
import os
import json
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO

# --- CONFIGURACIÓN ---
app = Flask(__name__)
AUTOS_FOLDER = 'static/autos' # Guardamos en static para que HTML pueda verlas
os.makedirs(AUTOS_FOLDER, exist_ok=True)

# Variables globales para control
camera = None
current_source = "0"  # Por defecto webcam
is_running = False
detected_plates_data = [] # Memoria temporal para el frontend
saved_plates_set = set()  # Para evitar duplicados en disco

# --- MODELOS (Asegúrate de tenerlos en la misma carpeta) ---
print("Cargando modelos IA...")
# Asumiendo: 0=Placa, 1=Auto (Ajusta según tu data.yaml)
CLASS_ID_PLATE = 0 
CLASS_ID_CAR = 1
main_model = YOLO("best.pt")
char_model = YOLO("best_char.pt") 

def is_plate_inside_car(plate_box, car_box):
    """Lógica de contención espacial"""
    px1, py1, px2, py2 = plate_box
    cx1, cy1, cx2, cy2 = car_box
    p_cx, p_cy = (px1 + px2) / 2, (py1 + py2) / 2
    return (cx1 < p_cx < cx2) and (cy1 < p_cy < cy2)

def generate_frames():
    global camera, current_source
    
    # Manejo de fuente de video (IP, Video o WebCam)
    source = int(current_source) if current_source.isdigit() else current_source
    cap = cv2.VideoCapture(source)
    
    track_history = {}
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        # 1. DETECCIÓN GENERAL
        results = main_model.track(frame, persist=True, verbose=False, conf=0.5)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.int().cpu().tolist()
            ids = results[0].boxes.id.int().cpu().tolist()

            cars = []
            plates = []

            for box, cls, trk_id in zip(boxes, classes, ids):
                if cls == CLASS_ID_CAR:
                    cars.append({'box': box, 'id': trk_id})
                elif cls == CLASS_ID_PLATE:
                    plates.append({'box': box, 'id': trk_id})

            # 2. VALIDACIÓN Y PROCESAMIENTO
            for plate in plates:
                valid_plate = False
                parent_car = None

                # Validar si placa está dentro de un auto
                for car in cars:
                    if is_plate_inside_car(plate['box'], car['box']):
                        valid_plate = True
                        parent_car = car
                        break
                
                px1, py1, px2, py2 = map(int, plate['box'])

                if valid_plate and parent_car:
                    # Dibujar linea de asociación
                    cx1, cy1, cx2, cy2 = map(int, parent_car['box'])
                    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (100, 100, 100), 1) # Auto
                    cv2.line(frame, (int((px1+px2)/2), int((py1+py2)/2)), 
                             (int((cx1+cx2)/2), cy1), (0, 255, 255), 1)

                    p_id = plate['id']
                    
                    # Lógica de lectura o memoria
                    if p_id in track_history:
                        text = track_history[p_id]
                        color = (0, 255, 0)
                    else:
                        # Leer caracteres (OCR propio)
                        h, w, _ = frame.shape
                        crop = frame[max(0, py1-5):min(h, py2+5), max(0, px1-5):min(w, px2+5)]
                        char_res = char_model(crop, verbose=False, conf=0.5)
                        
                        plate_text = ""
                        if len(char_res[0].boxes) > 0:
                            c_boxes = char_res[0].boxes.xyxy.cpu().numpy()
                            c_cls = char_res[0].boxes.cls.int().cpu().tolist()
                            sorted_chars = sorted(zip(c_boxes, c_cls), key=lambda c: c[0][0])
                            
                            for _, cls_id in sorted_chars:
                                if cls_id < len(char_model.names):
                                    plate_text += char_model.names[cls_id]

                        # Validar longitud (6 a 8 chars)
                        if 6 <= len(plate_text) <= 8:
                            track_history[p_id] = plate_text
                            text = plate_text
                            color = (0, 255, 0)

                            # --- GUARDAR EVIDENCIA ---
                            if plate_text not in saved_plates_set:
                                saved_plates_set.add(plate_text)
                                
                                # Recortar el AUTO completo (no solo la placa)
                                car_img = frame[max(0, cy1):min(frame.shape[0], cy2), 
                                                max(0, cx1):min(frame.shape[1], cx2)]
                                
                                filename = f"{plate_text}_{int(time.time())}.jpg"
                                filepath = os.path.join(AUTOS_FOLDER, filename)
                                cv2.imwrite(filepath, car_img)
                                
                                # Enviar datos a la lista web
                                detected_plates_data.insert(0, {
                                    "plate": plate_text,
                                    "image": f"static/autos/{filename}",
                                    "time": time.strftime("%H:%M:%S")
                                })
                        else:
                            text = "..."
                            color = (0, 165, 255)

                    cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
                    cv2.putText(frame, text, (px1, py1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Codificar frame para web
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_config', methods=['POST'])
def update_config():
    global current_source
    data = request.json
    current_source = data.get('source', '0')
    # Nota: En producción, necesitarías reiniciar el thread del generador correctamente
    return jsonify({"status": "updated", "source": current_source})

@app.route('/get_data')
def get_data():
    return jsonify(detected_plates_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
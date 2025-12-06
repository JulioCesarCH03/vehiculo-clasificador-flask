import os
from flask import Flask, render_template, jsonify, send_file
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO

app = Flask(__name__)

# URLs de imágenes (igual que en tu código original)
URLS_IMAGENES = {
    "buses": "https://raw.githubusercontent.com/JulioCesarCH03/dataset-vehiculos-clasificador/main/buses.jpg",
    "camiones": "https://raw.githubusercontent.com/JulioCesarCH03/dataset-vehiculos-clasificador/main/camiones.jpg",
    "hatchback": "https://raw.githubusercontent.com/JulioCesarCH03/dataset-vehiculos-clasificador/main/hatchback.jpg",
    "sedan": "https://raw.githubusercontent.com/JulioCesarCH03/dataset-vehiculos-clasificador/main/sedan.jpg",
    "suv": "https://raw.githubusercontent.com/JulioCesarCH03/dataset-vehiculos-clasificador/main/suv.jpg"
}

# Lista ordenada de tipos
TIPOS = list(URLS_IMAGENES.keys())

def descargar_imagen(url):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert('RGB')
        return np.array(img)
    except Exception as e:
        print(f"Error descargando {url}: {e}")
        return None

def procesar_canny(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    canny_rgb = np.zeros_like(img_rgb)
    canny_rgb[edges > 0] = [0, 255, 0]
    return canny_rgb

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data/<int:index>')
def get_data(index):
    if index >= len(TIPOS):
        return jsonify({"done": True})
    tipo = TIPOS[index]
    url = URLS_IMAGENES[tipo]
    return jsonify({
        "done": False,
        "index": index,
        "tipo": tipo.upper(),
        "url": url
    })

@app.route('/canny/<int:index>')
def get_canny(index):
    if index >= len(TIPOS):
        return "Índice inválido", 400
    url = URLS_IMAGENES[TIPOS[index]]
    img = descargar_imagen(url)
    if img is None:
        return "Error al cargar imagen", 500
    canny_img = procesar_canny(img)
    pil_img = Image.fromarray(canny_img)
    buf = BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

# 🔑 ¡Este bloque es clave para Render y desarrollo local!
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
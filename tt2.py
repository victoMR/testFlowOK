from PIL import Image
import numpy as np
from pix2tex.cli import LatexOCR
from pymongo import MongoClient
import re
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple
from datetime import datetime

# Conexión a MongoDB
MONGODB_URI = "mongodb+srv://admin:admin@testflowclouster.wxv1v.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(MONGODB_URI)
db = client["testflow"]
collection = db["formulas"]


def detect_formula_regions_debug(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    # Convertir a escala de grises si no lo está
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Aplicar threshold adaptativo para separar el texto del fondo
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Dilatación para conectar componentes cercanos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 15))  # Aumentado para capturar fórmulas completas
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Encontrar contornos
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ajustes de filtro de tamaño
    min_width, min_height = 200, 80  # Aumentado para fórmulas más completas
    max_width, max_height = 600, 300  # Tamaño máximo
    padding = 20  # Aumentado para capturar áreas completas

    formula_regions = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filtrar por tamaño de contorno
        if min_width < w < max_width and min_height < h < max_height:
            # Añadir padding para capturar el área completa de la fórmula
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            formula_regions.append((x, y, w, h))

    # Ordenar regiones de arriba a abajo
    formula_regions.sort(key=lambda x: x[1])

    # Visualización de regiones detectadas
    debug_image = image.copy()
    for i, (x, y, w, h) in enumerate(formula_regions):
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(debug_image, f"R{i + 1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    plt.figure(figsize=(10, 8))
    plt.imshow(debug_image, cmap='gray')
    plt.title("Regiones de Fórmulas Detectadas")
    plt.show()

    return formula_regions


def save_to_database(latex_formula: str, problem_type: str, difficulty: str):
    # Fecha y hora de escaneo
    scan_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    problem = {
        "latex_formula": latex_formula,
        "problem_type": problem_type,
        "difficulty": difficulty,
        "usage_count": 0,
        "scan_date": scan_date  # Agregar fecha de escaneo
    }

    # Verificar si la fórmula ya existe en la base de datos
    existing_problem = collection.find_one({"latex_formula": latex_formula})

    if existing_problem:
        # Si ya existe, solo incrementa el contador de uso
        collection.update_one({"latex_formula": latex_formula}, {"$inc": {"usage_count": 1}})
    else:
        # Si no existe, inserta la fórmula como un nuevo documento
        collection.insert_one(problem)


def classify_problem_type(latex: str) -> str:
    if re.search(r"\\frac", latex):
        return "Álgebra"
    elif re.search(r"\\int", latex):
        return "Cálculo"
    elif re.search(r"\\sum", latex):
        return "Series y sumas"
    elif re.search(r"\\theta", latex):
        return "Geometría"
    else:
        return "Otro"


def classify_difficulty(latex: str) -> str:
    if len(latex) > 200:
        return "Difícil"
    elif len(latex) > 100:
        return "Moderado"
    else:
        return "Fácil"


def process_formula_region(image: Image.Image, region: Tuple[int, int, int, int], model: LatexOCR) -> dict:
    x, y, w, h = region
    formula_image = image.crop((x, y, x + w, y + h))

    try:
        latex_formula = model(formula_image)
        problem_type = classify_problem_type(latex_formula)
        difficulty = classify_difficulty(latex_formula)

        # Guardar la fórmula detectada en la base de datos
        save_to_database(latex_formula, problem_type, difficulty)

        return {
            "latex": latex_formula,
            "type": problem_type,
            "difficulty": difficulty,
            "position": (x, y, w, h)
        }
    except Exception as e:
        print(f"Error procesando región {region}: {str(e)}")
        return None


def main():
    image_path = "img2.jpg"
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    # Detectar regiones de fórmulas con depuración visual
    formula_regions = detect_formula_regions_debug(image_np)

    model = LatexOCR()
    results = []
    print(f"Se encontraron {len(formula_regions)} fórmulas potenciales")

    for i, region in enumerate(formula_regions, 1):
        print(f"\nProcesando fórmula {i}/{len(formula_regions)}")
        result = process_formula_region(image, region, model)
        if result:
            results.append(result)
            print(f"Fórmula {i}:")
            print(f"LaTeX: {result['latex']}")
            print(f"Tipo: {result['type']}")
            print(f"Dificultad: {result['difficulty']}")

    print(f"\nSe procesaron exitosamente {len(results)} fórmulas")


if __name__ == "__main__":
    main()

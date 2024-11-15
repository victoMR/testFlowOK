from PIL import Image
from pix2tex.cli import LatexOCR
from pymongo import MongoClient
import re

# Conexión a MongoDB
MONGODB_URI = "mongodb+srv://admin:admin@testflowclouster.wxv1v.mongodb.net/?retryWrites=true&w=majority"

# Conexión a la base de datos y la colección
client = MongoClient(MONGODB_URI)
db = client["testflow"]
collection = db["formulas"]

# Función para guardar en la base de datos
def save_to_database(latex_formula, problem_type, difficulty):
    # Creando un documento con la fórmula, tipo y dificultad
    problem = {
        "latex_formula": latex_formula,
        "problem_type": problem_type,
        "difficulty": difficulty,
        "usage_count": 0  # Inicializamos el contador de uso
    }
    # Insertamos el problema si no existe, si ya existe incrementamos el contador
    existing_problem = collection.find_one({"latex_formula": latex_formula})
    if existing_problem:
        collection.update_one({"latex_formula": latex_formula}, {"$inc": {"usage_count": 1}})
    else:
        collection.insert_one(problem)

# Función para clasificar el tipo de problema basado en el LaTeX
def classify_problem_type(latex):
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

# Función para clasificar la dificultad basada en la longitud del LaTeX
def classify_difficulty(latex):
    if len(latex) > 200:
        return "Difícil"
    elif len(latex) > 100:
        return "Moderado"
    else:
        return "Fácil"

# Función para cargar y preprocesar la imagen
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

# Función principal para cargar la imagen, procesarla y obtener LaTeX
def main():
    image_path = "img.png"  # Cambia esto a la ruta de tu imagen
    image = preprocess_image(image_path)

    # Inicializamos el modelo de pix2tex
    model = LatexOCR()

    # Hacemos la predicción de LaTeX
    latex_formula = model(image)
    print("Fórmula LaTeX reconocida:", latex_formula)

    # Clasificamos el tipo y la dificultad del problema
    problem_type = classify_problem_type(latex_formula)
    difficulty = classify_difficulty(latex_formula)

    # Guardamos el resultado en la base de datos
    save_to_database(latex_formula, problem_type, difficulty)

    print(f"Tipo de problema: {problem_type}")
    print(f"Dificultad: {difficulty}")

# Ejecutamos la función principal
if __name__ == "__main__":
    main()

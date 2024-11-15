import torch
import json
from pathlib import Path
from math_extractor import MathFormulaExtractor, DebugVisualizer
from image_processor import ImagePreprocessor


def load_vocab(vocab_path: str):
    """Carga el vocabulario desde el archivo JSON."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab["idx_to_token"]


def decode_output(output: torch.Tensor, idx_to_token: dict) -> str:
    """Decodifica la salida del modelo a una fórmula LaTeX."""
    # Asegurar que output sea un tensor 1D
    if isinstance(output, torch.Tensor):
        indices = output.squeeze().cpu().numpy()
        if indices.ndim == 0:  # Si es un escalar
            indices = [indices.item()]
    else:
        indices = output
    
    # Construir la fórmula
    latex = "$"  # Iniciar modo matemático
    current_command = ""
    
    for idx in indices:
        # Convertir idx a string para buscar en el diccionario
        token = idx_to_token[str(int(idx))]
        
        # Ignorar tokens especiales
        if token in ["SOS", "EOS", "PAD"]:
            continue
            
        if token.startswith("\\"):
            # Comandos LaTeX
            if current_command:
                latex += current_command + " "
            current_command = token
        elif token in ["{", "}"]:
            # Manejar llaves
            if current_command:
                latex += current_command
                current_command = ""
            latex += token
        elif token in ["_", "^"]:
            # Manejar subíndices y superíndices
            if current_command:
                latex += current_command
                current_command = ""
            latex += token
        elif token in ["=", "+", "-", "*", "/", "(", ")", "[", "]"]:
            # Símbolos especiales
            if current_command:
                latex += current_command + " "
                current_command = ""
            latex += token
        elif token.strip():  # Ignorar espacios en blanco
            # Otros caracteres
            if current_command:
                latex += current_command + " "
                current_command = ""
            latex += token
    
    # Añadir cualquier comando pendiente
    if current_command:
        latex += current_command
    
    latex += "$"  # Cerrar modo matemático
    
    # Limpiar la fórmula
    latex = latex.replace("  ", " ").strip()
    return latex


def main():
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug_dir = Path("debug")
    debug_visualizer = DebugVisualizer(debug_dir)
    
    # Cargar vocabulario
    vocab_path = "vocab.json"
    idx_to_token = load_vocab(vocab_path)
    vocab_size = len(idx_to_token)
    
    # Inicializar modelo
    model = MathFormulaExtractor(vocab_size=vocab_size).to(device)
    preprocessor = ImagePreprocessor()
    
    # Cargar imagen
    image_path = "img2.jpg"
    try:
        image_tensor, structure = preprocessor.preprocess(
            image_path,
            debug_visualizer=debug_visualizer
        )
        image_tensor = image_tensor.to(device)
        
        # Mostrar información sobre los componentes detectados
        print("\nComponentes detectados:")
        for comp_type, components in structure.items():
            if components:  # Solo mostrar si hay componentes de este tipo
                print(f"{comp_type}: {len(components)} componentes")
                if comp_type == 'fraction_line':
                    for i, frac in enumerate(components):
                        print(f"  Fracción {i+1}:")
                        print(f"    Numerador: {len(frac['numerator'])} componentes")
                        print(f"    Denominador: {len(frac['denominator'])} componentes")
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Inferencia
    model.eval()
    try:
        with torch.no_grad():
            output = model(image_tensor)
            print(f"\nOutput shape: {output.shape if isinstance(output, torch.Tensor) else 'Not a tensor'}")
            
            # Decodificar y mostrar resultado
            latex = decode_output(output, idx_to_token)
            print(f"LaTeX Formula: {latex}")
            
            # Guardar resultado en el log
            debug_visualizer.save_step(None, "result", info=f"LaTeX: {latex}")
    except Exception as e:
        print(f"Error durante la inferencia: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

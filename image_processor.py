import cv2
import numpy as np
from PIL import Image, ImageEnhance
from torchvision import transforms
from typing import List, Tuple, Dict
import torch

class ImagePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
    def detect_components(self, binary_image: np.ndarray) -> List[Dict]:
        """Detecta y clasifica componentes individuales de la fórmula."""
        components = []
        
        # Encontrar todos los componentes conectados
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_image, connectivity=8
        )
        
        # Crear imagen de debug
        debug_image = cv2.cvtColor(binary_image.copy(), cv2.COLOR_GRAY2BGR)
        
        # Filtrar el fondo (label 0)
        for label in range(1, num_labels):
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]
            area = stats[label, cv2.CC_STAT_AREA]
            
            # Filtrar componentes muy pequeños
            if area < 15 or w < 3 or h < 3:  # Ajustado para detectar puntos y componentes pequeños
                continue
                
            # Extraer el componente
            component = binary_image[y:y+h, x:x+w]
            
            # Clasificar el tipo de componente
            component_type = self._classify_component(component, w/h)
            
            # Guardar el componente
            components.append({
                'bbox': (x, y, w, h),
                'image': component,
                'type': component_type,
                'centroid': centroids[label],
                'area': area
            })
            
            # Dibujar en la imagen de debug
            color = {
                'fraction_line': (0, 255, 0),
                'horizontal_line': (0, 255, 255),
                'vertical_line': (255, 0, 0),
                'operator': (255, 0, 255),
                'letter': (0, 255, 0),
                'number': (255, 128, 0),
                'symbol': (0, 0, 255),
                'unknown': (128, 128, 128)
            }.get(component_type, (128, 128, 128))
            
            cv2.rectangle(debug_image, (x, y), (x+w, y+h), color, 1)
            cv2.putText(debug_image, component_type, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Guardar imagen de debug si hay un visualizador disponible
        if hasattr(self, 'debug_visualizer') and self.debug_visualizer is not None:
            self.debug_visualizer.save_step(debug_image, 'component_detection')
        
        # Ordenar componentes de izquierda a derecha y de arriba a abajo
        components.sort(key=lambda c: (c['centroid'][1] // 30, c['centroid'][0]))
        
        return components
    
    def _classify_component(self, component: np.ndarray, aspect_ratio: float) -> str:
        """Clasifica el tipo de componente basado en sus características."""
        h, w = component.shape
        area = h * w
        
        # Características para clasificación
        density = np.sum(component > 0) / area
        
        # Obtener contornos para calcular características adicionales
        contours, _ = cv2.findContours(
            component.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return "unknown"
        
        # Calcular características más detalladas
        perimeter = cv2.arcLength(contours[0], True)
        compactness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Momentos de Hu para características de forma
        moments = cv2.moments(component)
        hu_moments = cv2.HuMoments(moments) if moments['m00'] != 0 else np.zeros(7)
        
        # Características adicionales
        extent = area / (w * h)  # Área relativa al rectángulo delimitador
        solidity = area / cv2.contourArea(contours[0]) if cv2.contourArea(contours[0]) > 0 else 0
        
        # Imprimir características para debugging
        print(f"Component stats: size={w}x{h}, aspect_ratio={aspect_ratio:.2f}, "
              f"density={density:.2f}, compactness={compactness:.2f}, "
              f"extent={extent:.2f}, solidity={solidity:.2f}")
        
        # Clasificación basada en reglas ajustadas
        # Líneas horizontales (fracciones y símbolos de igual)
        if aspect_ratio > 2.5:
            if density > 0.4 and h <= 5:  # Líneas de fracción son generalmente delgadas
                return "fraction_line"
            if density > 0.3:
                return "horizontal_line"
        
        # Líneas verticales y barras
        if aspect_ratio < 0.4:
            if density > 0.5:
                return "vertical_line"
            return "operator"
        
        # Operadores matemáticos
        if 0.7 < aspect_ratio < 1.3:  # Forma aproximadamente cuadrada
            if area < 400:  # Operadores suelen ser pequeños
                if density < 0.3:
                    return "operator"
                if density < 0.5 and compactness > 0.6:
                    return "operator"
        
        # Números
        if 0.4 < aspect_ratio < 2.0:
            if 0.4 < density < 0.7:
                if compactness > 0.5 and solidity > 0.8:
                    if extent > 0.6:
                        return "number"
        
        # Letras
        if 0.5 < aspect_ratio < 2.0:
            if 0.2 < density < 0.6:
                if compactness > 0.3:
                    if extent > 0.4:
                        return "letter"
        
        # Símbolos matemáticos especiales
        if density > 0.15:
            if compactness < 0.8:
                if solidity < 0.9:
                    return "symbol"
        
        # Si no se ajusta a ninguna categoría anterior, clasificar basado en densidad
        if density < 0.3:
            return "operator"
        if density < 0.5:
            return "letter"
        if density < 0.7:
            return "number"
        
        return "symbol"
    
    def analyze_structure(self, components: List[Dict]) -> Dict:
        """Analiza la estructura de la fórmula basada en los componentes."""
        structure = {
            'fraction_line': [],
            'horizontal_line': [],
            'vertical_line': [],
            'operator': [],
            'letter': [],
            'number': [],
            'symbol': [],
            'subscript': [],
            'superscript': []
        }
        
        # Primero, agrupar componentes por tipo
        for comp in components:
            structure[comp['type']].append(comp)
        
        # Analizar fracciones
        for line in structure['fraction_line']:
            x, y, w, h = line['bbox']
            line_y = y + h/2
            
            # Encontrar componentes arriba y abajo de la línea
            numerator = []
            denominator = []
            
            for comp_type in ['letter', 'number', 'operator', 'symbol']:
                for comp in structure[comp_type]:
                    comp_x, comp_y, comp_w, comp_h = comp['bbox']
                    comp_center_y = comp_y + comp_h/2
                    
                    # Verificar si el componente está alineado horizontalmente con la línea
                    if comp_x + comp_w >= x and comp_x <= x + w:
                        if comp_center_y < line_y:
                            numerator.append(comp)
                        elif comp_center_y > line_y:
                            denominator.append(comp)
            
            # Ordenar componentes de izquierda a derecha
            numerator.sort(key=lambda c: c['bbox'][0])
            denominator.sort(key=lambda c: c['bbox'][0])
            
            # Actualizar la estructura de la fracción
            line['numerator'] = numerator
            line['denominator'] = denominator
        
        # Detectar subíndices y superíndices
        for comp_type in ['letter', 'number']:
            for i, base_comp in enumerate(structure[comp_type]):
                base_x, base_y, base_w, base_h = base_comp['bbox']
                base_center_y = base_y + base_h/2
                
                # Buscar componentes cercanos que podrían ser sub/superíndices
                for other_comp in components:
                    if other_comp == base_comp:
                        continue
                        
                    other_x, other_y, other_w, other_h = other_comp['bbox']
                    other_center_y = other_y + other_h/2
                    
                    # Verificar si está cerca horizontalmente
                    if other_x > base_x + base_w/2 and other_x < base_x + base_w * 2:
                        # Verificar posición vertical
                        if other_center_y < base_center_y - base_h/4:
                            structure['superscript'].append({
                                'base': base_comp,
                                'script': other_comp
                            })
                        elif other_center_y > base_center_y + base_h/4:
                            structure['subscript'].append({
                                'base': base_comp,
                                'script': other_comp
                            })
        
        return structure
    
    def preprocess(self, image_path: str, debug_visualizer=None) -> Tuple[torch.Tensor, Dict]:
        """Preprocesa la imagen y retorna el tensor y la estructura de la fórmula."""
        # Cargar imagen
        image = Image.open(image_path).convert('L')
        if debug_visualizer:
            debug_visualizer.save_step(np.array(image), 'original')
        
        # Mejorar contraste
        image = ImageEnhance.Contrast(image).enhance(2.0)
        
        # Binarización adaptativa
        image_np = np.array(image)
        binary = cv2.adaptiveThreshold(
            image_np,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        if debug_visualizer:
            debug_visualizer.save_step(binary, 'binary')
        
        # Eliminar ruido
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        if debug_visualizer:
            debug_visualizer.save_step(cleaned, 'cleaned')
        
        # Detectar región de la fórmula
        contours, _ = cv2.findContours(
            cleaned,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)
            pad = 10
            x = max(0, x-pad)
            y = max(0, y-pad)
            w = min(cleaned.shape[1]-x, w+2*pad)
            h = min(cleaned.shape[0]-y, h+2*pad)
            formula_region = cleaned[y:y+h, x:x+w]
        else:
            formula_region = cleaned
            
        # Detectar y analizar componentes
        components = self.detect_components(formula_region)
        structure = self.analyze_structure(components)
        
        if debug_visualizer:
            # Imagen original con componentes detectados
            debug_image = cv2.cvtColor(formula_region, cv2.COLOR_GRAY2BGR)
            
            # Crear directorios para cada tipo de componente
            component_types = ['fraction_line', 'horizontal_line', 'vertical_line', 
                             'operator', 'letter', 'number', 'symbol', 'unknown']
            for comp_type in component_types:
                (debug_visualizer.debug_dir / 'components' / comp_type).mkdir(parents=True, exist_ok=True)
            
            # Dibujar y etiquetar cada componente
            for comp in components:
                x, y, w, h = comp['bbox']
                color = {
                    'fraction_line': (0, 255, 0),    # Verde
                    'horizontal_line': (0, 255, 255), # Amarillo
                    'vertical_line': (255, 0, 0),    # Azul
                    'operator': (255, 0, 255),       # Magenta
                    'letter': (0, 128, 0),           # Verde oscuro
                    'number': (255, 128, 0),         # Naranja
                    'symbol': (0, 0, 255),           # Rojo
                    'unknown': (128, 128, 128)       # Gris
                }.get(comp['type'], (128, 128, 128))
                
                # Dibujar rectángulo y etiqueta
                cv2.rectangle(debug_image, (x, y), (x+w, y+h), color, 1)
                cv2.putText(
                    debug_image,
                    f"{comp['type']}",
                    (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    color,
                    1
                )
                
                # Guardar componente individual
                comp_debug = cv2.cvtColor(comp['image'], cv2.COLOR_GRAY2BGR)
                debug_visualizer.save_step(
                    comp_debug,
                    f'components/{comp["type"]}',
                    f"pos: ({x},{y}), size: {w}x{h}"
                )
            
            debug_visualizer.save_step(debug_image, 'detection', 'Componentes detectados')
        
        # Convertir a tensor
        image = Image.fromarray(formula_region)
        tensor = self.transform(image).unsqueeze(0)
        
        return tensor, structure 
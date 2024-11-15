import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import numpy as np
import cv2
import logging
from pathlib import Path

class DebugVisualizer:
    def __init__(self, debug_dir: str):
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear directorio para componentes
        self.components_dir = self.debug_dir / 'components'
        self.components_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar logging
        logging.basicConfig(
            filename=self.debug_dir / 'process.log',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def save_step(self, image: np.ndarray, step_name: str, info: str = ""):
        """Guarda una imagen de un paso del proceso."""
        # Manejar rutas anidadas
        if '/' in step_name or '\\' in step_name:
            step_parts = Path(step_name).parts
            step_dir = self.debug_dir.joinpath(*step_parts)
        else:
            step_dir = self.debug_dir / step_name
            
        # Crear directorio con todos los padres necesarios
        step_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar información en el log
        logging.info(f"Paso {step_name}: {info}")
        
        # Si no hay imagen, solo guardar la información
        if image is None:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = step_dir / f"{timestamp}.png"
        
        # Normalizar imagen si es necesario
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
            
        cv2.imwrite(str(filepath), image)

class CNNFeatureExtractor(nn.Module):
    """Extractor de características CNN basado en ResNet modificado."""
    def __init__(self, output_channels=512):
        super().__init__()
        
        # Capas convolucionales para extracción de características
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Bloques residuales
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4)
        self.layer3 = self._make_layer(256, output_channels, 6)
        
    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(blocks-1):
            layers.extend([
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        return x

class RowEncoder(nn.Module):
    """Codificador RNN para procesar filas de características."""
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
    
    def forward(self, x):
        # x shape: (batch_size * H, W, channels)
        output, _ = self.rnn(x)
        return output

class Attention(nn.Module):
    """Mecanismo de atención para el decodificador."""
    def __init__(self, encoder_dim, decoder_dim):
        super().__init__()
        self.attention = nn.Linear(encoder_dim + decoder_dim, decoder_dim)
        self.v = nn.Parameter(torch.rand(decoder_dim))
        
    def forward(self, hidden, encoder_outputs):
        # Calcular scores de atención
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attention(
            torch.cat((hidden, encoder_outputs), dim=2)
        ))
        
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        attention = torch.bmm(energy, v.transpose(1, 2))
        
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    """Decodificador RNN con atención."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, encoder_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(encoder_dim, hidden_dim)
        self.rnn = nn.GRU(
            embed_dim + encoder_dim,
            hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden, encoder_outputs):
        # Embedding
        embedded = self.embedding(x)
        
        # Atención
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        context = torch.bmm(attn_weights, encoder_outputs)
        
        # Combinar embedding y contexto
        rnn_input = torch.cat((embedded, context), dim=2)
        
        # RNN
        output, hidden = self.rnn(rnn_input, hidden)
        
        # Predicción
        prediction = self.fc(output)
        
        return prediction, hidden, attn_weights

class MathFormulaExtractor(nn.Module):
    """Modelo completo para extracción de fórmulas matemáticas."""
    def __init__(self, vocab_size, hidden_dim=512):
        super().__init__()
        self.cnn = CNNFeatureExtractor(output_channels=hidden_dim)
        self.row_encoder = RowEncoder(hidden_dim, hidden_dim)
        self.decoder = Decoder(
            vocab_size=vocab_size,
            embed_dim=hidden_dim//2,
            hidden_dim=hidden_dim,
            encoder_dim=hidden_dim*2  # Bidirectional
        )
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
    def forward(self, image, target=None, teacher_forcing_ratio=0.5):
        # Extraer características
        features = self.cnn(image)
        
        # Preparar para codificación por filas
        B, C, H, W = features.size()
        features = features.permute(0, 2, 3, 1)
        features = features.reshape(B*H, W, C)
        
        # Codificar filas
        encoded = self.row_encoder(features)
        encoded = encoded.reshape(B, H*W, -1)
        
        if self.training and target is not None:
            return self._train_decode(encoded, target, teacher_forcing_ratio)
        else:
            return self._beam_search(encoded)

    def _train_decode(self, encoded, target, teacher_forcing_ratio):
        batch_size = encoded.size(0)
        max_length = target.size(1)
        
        # Inicializar salidas
        outputs = torch.zeros(batch_size, max_length, self.vocab_size).to(encoded.device)
        
        # Inicializar entrada con token SOS
        decoder_input = torch.ones(batch_size, 1).long().to(encoded.device)
        decoder_hidden = torch.zeros(1, batch_size, self.hidden_dim).to(encoded.device)
        
        for t in range(max_length):
            decoder_output, decoder_hidden, _ = self.decoder(
                decoder_input, decoder_hidden, encoded
            )
            
            outputs[:, t:t+1] = decoder_output
            
            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            decoder_input = target[:, t:t+1] if teacher_force else decoder_output.argmax(2)
        
        return outputs

    def _beam_search(self, encoded, beam_width=5, max_length=150):
        batch_size = encoded.size(0)
        device = encoded.device

        # Inicializar con token SOS
        start_tokens = torch.ones(batch_size, 1).long().to(device)
        hidden = torch.zeros(1, batch_size, self.hidden_dim).to(device)

        # Inicializar beam search
        sequences = [(start_tokens, hidden, 0.0)]
        completed_sequences = []
        completed_scores = []

        for _ in range(max_length):
            candidates = []

            for seq, hidden, score in sequences:
                if seq[0, -1].item() == 1:  # EOS token
                    completed_sequences.append((seq, score))
                    continue

                # Obtener predicciones
                output, new_hidden, _ = self.decoder(seq[:, -1:], hidden, encoded)
                logits = output.squeeze(1)
                log_probs = F.log_softmax(logits, dim=-1)

                # Obtener top-k candidatos
                values, indices = log_probs.topk(beam_width)
                
                for i in range(beam_width):
                    token = indices[:, i:i+1]
                    new_seq = torch.cat([seq, token], dim=1)
                    new_score = score + values[:, i].item()
                    candidates.append((new_seq, new_hidden, new_score))

            # Ordenar y seleccionar los mejores candidatos
            candidates.sort(key=lambda x: x[2], reverse=True)
            sequences = candidates[:beam_width]

            # Verificar si todos los beams han terminado
            if all(seq[0, -1].item() == 1 for seq, _, _ in sequences):
                break

        # Seleccionar la mejor secuencia
        if completed_sequences:
            completed_sequences.sort(key=lambda x: x[1], reverse=True)
            best_sequence = completed_sequences[0][0]
        else:
            best_sequence = sequences[0][0]

        # Convertir a logits para mantener consistencia con la interfaz
        output = torch.zeros(best_sequence.size(1), self.vocab_size, device=device)
        for i, idx in enumerate(best_sequence[0]):
            output[i, idx] = 1.0

        return output
        
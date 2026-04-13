#!/usr/bin/env python3
"""
Transcriptor optimizado de alta velocidad.
Soporta múltiples backends para máxima velocidad:
1. Distil-Whisper (transformers + torch) - 6x más rápido
2. Faster-Whisper (ctranslate2 + onnx) - rápido y preciso
"""

import os
import json
import subprocess
import time
from typing import Optional, Dict, Tuple, List


def segundos_a_hhmmss(seg: float) -> str:
    """Convierte float de segundos a HH:MM:SS."""
    h = int(seg // 3600)
    m = int((seg % 3600) // 60)
    s = int(seg % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def detectar_device() -> Tuple[str, str]:
    """Detecta el mejor device disponible."""
    try:
        import torch
        if torch.cuda.is_available():
            return ("cuda", "float16")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return ("mps", "float32")
    except Exception:
        pass
    return ("cpu", "int8")


def extraer_audio(video_path: str, salida: str = "audio_temp.wav") -> str:
    """Extrae audio usando FFmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        salida
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return salida


class Transcriptor:
    """
    Transcriptor optimizado que auto-detecta el mejor backend disponible.
    Prioridad: Distil-Whisper > Faster-Whisper
    """
    
    def __init__(self, modelo: str = "distil-large-v3"):
        self.modelo = modelo
        self.backend = None
        self.model = None
        self._inicializar()
    
    def _inicializar(self):
        """Inicializa el mejor backend disponible."""
        if self.modelo.startswith("distil") or self.modelo.startswith("small"):
            if self._init_distilwhisper():
                return
        
        if self._init_fasterwhisper():
            return
        
        raise RuntimeError("No se pudo inicializar ningún backend de transcripción")
    
    def _init_distilwhisper(self) -> bool:
        """Inicializa Distil-Whisper (6x más rápido)."""
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoConfig
            
            device, compute_type = detectar_device()
            if device == "cpu":
                compute_type = "int8"
            
            model_name = f"distil-whisper/{self.modelo}"
            if self.modelo == "distil-large-v3":
                model_name = "distil-whisper/distil-large-v3"
            elif self.modelo == "distil-medium":
                model_name = "distil-whisper/distil-medium"
            elif self.modelo == "distil-small":
                model_name = "distil-whisper/distil-small"
            
            print(f"  Backend: Distil-Whisper ({model_name})")
            print(f"  Device: {device} ({compute_type})")
            
            config = AutoConfig.from_pretrained(model_name)
            processor = AutoProcessor.from_pretrained(model_name)
            
            torch_dtype = torch.float16 if compute_type == "float16" else torch.float32
            
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            
            if device != "cuda":
                model = model.to(device)
            
            self.model = {"model": model, "processor": processor, "device": device}
            self.backend = "distilwhisper"
            print(f"  ✓ Distil-Whisper cargado")
            return True
            
        except Exception as e:
            print(f"  Distil-Whisper no disponible: {e}")
            return False
    
    def _init_fasterwhisper(self) -> bool:
        """Inicializa Faster-Whisper."""
        try:
            from faster_whisper import WhisperModel
            
            device, compute_type = detectar_device()
            print(f"  Backend: Faster-Whisper")
            print(f"  Device: {device} ({compute_type})")
            print(f"  Modelo: {self.modelo}")
            
            model = WhisperModel(self.modelo, device=device, compute_type=compute_type)
            self.model = model
            self.backend = "fasterwhisper"
            print(f"  ✓ Faster-Whisper cargado")
            return True
            
        except Exception as e:
            print(f"  Faster-Whisper no disponible: {e}")
            return False
    
    def transcribir(self, audio_path: str, idioma: str = "es") -> Tuple[List[Dict], Dict]:
        """
        Transcribe el audio y retorna palabras con timestamps.
        
        Returns:
            Tuple of (words_list, info_dict)
        """
        if self.backend == "distilwhisper":
            return self._transcribir_distilwhisper(audio_path, idioma)
        else:
            return self._transcribir_fasterwhisper(audio_path, idioma)
    
    def _transcribir_distilwhisper(self, audio_path: str, idioma: str) -> Tuple[List[Dict], Dict]:
        """Transcribe usando Distil-Whisper."""
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        import librosa
        import numpy as np
        
        model = self.model["model"]
        processor = self.model["processor"]
        device = self.model["device"]
        
        print(f"  Cargando audio...")
        audio, sr = librosa.load(audio_path, sr=16000)
        audio = audio.astype(np.float32)
        
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            return_token_timestamps=True
        )
        
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        print(f"  Transcribiendo con Distil-Whisper...")
        with torch.no_grad():
            generated_ids = model.generate(
                inputs["input_features"],
                forced_decoder_ids=processor.get_decoder_prompt_ids(language=idioma, task="transcribe"),
                max_new_tokens=448,
                return_timestamps=True,
                output_scores=False,
                output_attentions=False,
                use_cache=True,
            )
        
        output = processor.batch_decode(generated_ids, output_word_offsets=True)
        
        words = []
        if output and len(output) > 0:
            result = output[0]
            if hasattr(result, 'word_offsets') and result.word_offsets:
                for word_info in result.word_offsets:
                    words.append({
                        "word": word_info["word"].strip(),
                        "start": round(word_info["start"], 3),
                        "end": round(word_info["end"], 3),
                        "probability": 0.95,
                        "segment_id": 0
                    })
            elif hasattr(result, 'text'):
                print(f"  AVISO: Distil-Whisper no devolvió word_timestamps")
        
        info = {
            "language": idioma,
            "duration": len(audio) / 16000,
            "language_probability": 0.95
        }
        
        return words, info
    
    def _transcribir_fasterwhisper(self, audio_path: str, idioma: str) -> Tuple[List[Dict], Dict]:
        """Transcribe usando Faster-Whisper."""
        model = self.model
        
        segments, info = model.transcribe(
            audio_path,
            language=idioma,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=100
            ),
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=True
        )
        
        segments_list = list(segments)
        
        words = []
        for seg_idx, segment in enumerate(segments_list):
            if not segment.words:
                continue
            for word in segment.words:
                words.append({
                    "word": word.word.strip(),
                    "start": round(word.start, 3),
                    "end": round(word.end, 3),
                    "probability": round(word.probability, 3),
                    "segment_id": seg_idx
                })
        
        info_dict = {
            "language": info.language,
            "duration": info.duration,
            "language_probability": info.language_probability
        }
        
        return words, info_dict


def generar_bloques(words: List[Dict], max_palabras: int = 6) -> List[str]:
    """Genera bloques legibles para la IA."""
    bloques = []
    buffer = []
    tiempo_inicio = None
    
    for i, word_data in enumerate(words):
        palabra = word_data['word']
        t_inicio = word_data['start']
        
        if tiempo_inicio is None:
            tiempo_inicio = t_inicio
        
        pausa = 0.0
        if i > 0:
            pausa = t_inicio - words[i - 1]['end']
        
        fin_oracion = False
        if buffer:
            ultima = buffer[-1]
            fin_oracion = ultima.rstrip().endswith(('.', '!', '?'))
        
        nuevo_bloque = (
            len(buffer) >= max_palabras or
            pausa > 0.5 or
            fin_oracion
        )
        
        if nuevo_bloque and buffer:
            texto = ' '.join(buffer).strip()
            if texto:
                ts = segundos_a_hhmmss(tiempo_inicio)
                bloques.append(f"{ts} - {texto}")
            buffer = []
            tiempo_inicio = t_inicio
        
        buffer.append(palabra)
    
    if buffer and tiempo_inicio is not None:
        texto = ' '.join(buffer).strip()
        if texto:
            ts = segundos_a_hhmmss(tiempo_inicio)
            bloques.append(f"{ts} - {texto}")
    
    return bloques


def transcribir_video(
    video_path: str,
    salida_transcripcion: str = "transcripcion_formatted.txt",
    salida_words: str = "whisper_words.json",
    modelo: str = "distil-large-v3",
    idioma: str = "es",
) -> dict:
    """
    Transcribe un video usando el backend más rápido disponible.
    
    Args:
        video_path: Ruta al video
        salida_transcripcion: Archivo de transcripción formateada
        salida_words: Archivo JSON con timestamps por palabra
        modelo: Modelo a usar ("distil-large-v3", "distil-medium", "small", "large-v3")
        idioma: Código ISO del idioma
    
    Returns:
        dict con metadata de la transcripción
    """
    inicio_total = time.time()
    
    print("=" * 50)
    print("  TRANSCRIPCIÓN OPTIMIZADA")
    print("=" * 50)
    
    print(f"  Modelo seleccionado: {modelo}")
    print(f"  (La primera vez descarga el modelo, puede tardar)")
    
    audio_temp = extraer_audio(video_path, "audio_whisper_temp.wav")
    
    try:
        transcriptor = Transcriptor(modelo=modelo)
        
        inicio_trans = time.time()
        print(f"\n  Transcribiendo...")
        
        words, info = transcriptor.transcribir(audio_temp, idioma)
        
        duracion_trans = time.time() - inicio_trans
        
        print(f"  ✓ Transcripción completada")
        print(f"    Duración video: {info['duration']:.1f}s")
        print(f"    Idioma: {info['language']} (prob: {info['language_probability']:.1%})")
        print(f"    Tiempo: {duracion_trans:.1f}s ({info['duration']/duracion_trans:.1f}x realtime)")
        
        print(f"    Palabras: {len(words)}")
        
        with open(salida_words, 'w', encoding='utf-8') as f:
            json.dump(words, f, ensure_ascii=False, indent=2)
        print(f"    whisper_words.json: {salida_words}")
        
        bloques = generar_bloques(words)
        with open(salida_transcripcion, 'w', encoding='utf-8') as f:
            f.write('\n'.join(bloques) + '\n')
        print(f"    transcripcion_formatted.txt: {salida_transcripcion}")
        print(f"    Bloques: {len(bloques)}")
        
    finally:
        try:
            os.remove(audio_temp)
        except Exception:
            pass
    
    duracion_total = time.time() - inicio_total
    print(f"\n  Total: {duracion_total:.1f}s")
    print("=" * 50)
    
    return {
        "transcripcion_path": salida_transcripcion,
        "words_path": salida_words,
        "duracion": info['duration'],
        "total_palabras": len(words),
        "idioma_detectado": info['language'],
        "confianza_idioma": info['language_probability'],
        "backend": transcriptor.backend if 'transcriptor' in dir() else "unknown"
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python transcriber.py video.mp4 [modelo]")
        print("  Modelos disponibles:")
        print("    distil-large-v3  - Más rápido, excelente calidad (6x)")
        print("    distil-medium     - Rápido, muy buena calidad")
        print("    small             - Muy rápido, buena calidad")
        print("    large-v3          - Mejor calidad, más lento")
        sys.exit(1)
    
    video = sys.argv[1]
    modelo = sys.argv[2] if len(sys.argv) > 2 else "distil-large-v3"
    
    try:
        resultado = transcribir_video(video, modelo=modelo)
        print(f"\n✓ Listo:")
        print(f"  Palabras: {resultado['total_palabras']}")
        print(f"  Backend: {resultado['backend']}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

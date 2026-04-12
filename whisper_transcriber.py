#!/usr/bin/env python3
"""
Transcripción automática con timestamps por palabra usando faster-whisper.
Genera transcripcion_formatted.txt con precisión de palabra individual.
"""

import os
import subprocess
from typing import List, Tuple, Optional


def segundos_a_hhmmss(seg: float) -> str:
    """Convierte float de segundos a HH:MM:SS."""
    h = int(seg // 3600)
    m = int((seg % 3600) // 60)
    s = int(seg % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def detectar_device() -> Tuple[str, str]:
    """Detecta si hay GPU disponible para Whisper."""
    try:
        import torch
        if torch.cuda.is_available():
            return ("cuda", "float16")
    except ImportError:
        pass
    return ("cpu", "int8")


def transcribir_video(
    video_path: str,
    salida_path: str = "transcripcion_formatted.txt",
    modelo: str = "large-v3",
    idioma: str = "es",
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
) -> str:
    """
    Transcribe un video usando faster-whisper y genera archivo con timestamps.
    
    Args:
        video_path: Ruta al video de entrada
        salida_path: Ruta del archivo de transcripción formateada
        modelo: Modelo de Whisper a usar ("tiny", "base", "small", "medium", "large-v3")
        idioma: Código de idioma ISO ("es", "en", etc.)
        device: "cpu" o "cuda" (None = auto-detectar)
        compute_type: "int8", "float16", etc. (None = auto-detectar)
    
    Returns:
        Ruta del archivo de transcripción generado
    """
    import time
    inicio_total = time.time()
    
    print("=" * 50)
    print("  TRANSCRIPCIÓN CON FASTER-WHISPER")
    print("=" * 50)
    
    if device is None or compute_type is None:
        device, compute_type = detectar_device()
        print(f"  Device: {device} ({compute_type})")
    
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("  ERROR: faster-whisper no instalado")
        print("  Instalar con: pip install faster-whisper")
        raise ImportError("faster-whisper no instalado")
    
    print(f"  Cargando modelo: {modelo}...")
    model = WhisperModel(modelo, device=device, compute_type=compute_type)
    
    audio_temp = "audio_temp_whisper.wav"
    
    print(f"  Extrayendo audio de: {video_path}")
    cmd_extract = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_temp
    ]
    result = subprocess.run(cmd_extract, capture_output=True)
    if result.returncode != 0:
        print(f"  ERROR extrayendo audio: {result.stderr.decode()[:200]}")
        return ""
    
    print(f"  Transcribiendo (esto puede tardar varios minutos)...")
    inicio_trans = time.time()
    
    segments, info = model.transcribe(
        audio_temp,
        language=idioma,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=200
        )
    )
    
    duracion_trans = time.time() - inicio_trans
    
    print(f"  ✓ Transcripción completada")
    print(f"    Duración video: {info.duration:.1f}s")
    print(f"    Idioma: {info.language} (prob: {info.language_probability:.1%})")
    print(f"    Tiempo: {duracion_trans:.1f}s")
    
    bloques: List[Tuple[float, str]] = []
    palabras_buffer: List[str] = []
    tiempo_inicio_bloque: Optional[float] = None
    
    for segment in segments:
        if not segment.words:
            continue
            
        for word in segment.words:
            palabra_texto = word.word.strip()
            if not palabra_texto:
                continue
            
            if tiempo_inicio_bloque is None:
                tiempo_inicio_bloque = word.start
            
            pausa = 0.4
            if palabras_buffer:
                ultima_palabra = segment.words[segment.words.index(word) - 1] if segment.words.index(word) > 0 else None
                if ultima_palabra:
                    pausa = word.start - ultima_palabra.end
            
            if pausa > 0.4 or len(palabras_buffer) >= 4:
                if palabras_buffer and tiempo_inicio_bloque is not None:
                    texto_bloque = ' '.join(palabras_buffer)
                    bloques.append((tiempo_inicio_bloque, texto_bloque))
                palabras_buffer = []
                tiempo_inicio_bloque = word.start
            
            palabras_buffer.append(palabra_texto)
    
    if palabras_buffer and tiempo_inicio_bloque is not None:
        texto_bloque = ' '.join(palabras_buffer)
        bloques.append((tiempo_inicio_bloque, texto_bloque))
    
    with open(salida_path, "w", encoding="utf-8") as f:
        for ts, texto in bloques:
            f.write(f"{segundos_a_hhmmss(ts)} - {texto}\n")
    
    palabras_totales = sum(len(seg.words) for seg in segments if seg.words)
    
    print(f"    Palabras transcritas: {palabras_totales}")
    print(f"    Bloques generados: {len(bloques)}")
    print(f"    Archivo: {salida_path}")
    
    try:
        os.remove(audio_temp)
    except:
        pass
    
    duracion_total = time.time() - inicio_total
    print(f"  Total: {duracion_total:.1f}s")
    print("=" * 50)
    
    return salida_path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python whisper_transcriber.py video.mp4 [salida.txt]")
        print("  Modelo por defecto: large-v3")
        print("  Idioma por defecto: español")
        sys.exit(1)
    
    video = sys.argv[1]
    salida = sys.argv[2] if len(sys.argv) > 2 else "transcripcion_formatted.txt"
    
    try:
        transcribir_video(video, salida)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

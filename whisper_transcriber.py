#!/usr/bin/env python3
"""
Transcripción de video para detección de clips virales.
Genera transcripcion_formatted.txt para que la IA analice el contenido.

Modos:
  A) Con transcripcion.txt manual → parser determinístico (instantáneo)
  B) Sin transcripción → faster-whisper large-v3-turbo (5-6 min)
"""

import os
import re
import subprocess
import time
from typing import List


def segundos_a_ms(seg: float) -> str:
    m = int(seg // 60)
    s = int(seg % 60)
    return f"{m}:{s:02d}"


def segundos_a_hhmmss(seg: float) -> str:
    h = int(seg // 3600)
    m = int((seg % 3600) // 60)
    s = int(seg % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def detectar_device() -> tuple:
    try:
        import torch
        if torch.cuda.is_available():
            return ("cuda", "float16")
    except Exception:
        pass
    return ("cpu", "int8")


def _extraer_audio(video_path: str) -> str:
    audio_temp = "audio_trans_temp.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        audio_temp
    ], capture_output=True, check=True)
    return audio_temp


def _remover_descripcion(texto: str) -> str:
    """
    Remueve descripción de tiempo al inicio del texto.
    Maneja texto pegado sin espacio.
    """
    patrones_desc = [
        re.compile(r'^\d+\s+minutos?\s+y\s+\d+\s+segundos?', re.IGNORECASE),
        re.compile(r'^\d+\s+minutos?(?!\s+y)', re.IGNORECASE),
        re.compile(r'^\d+\s+segundos?', re.IGNORECASE),
    ]
    for patron in patrones_desc:
        m = patron.match(texto)
        if m:
            resto = texto[len(m.group(0)):]
            return resto.lstrip()
    return texto


def parsear_transcripcion_youtube(
    ruta: str,
    salida: str = "transcripcion_formatted.txt"
) -> str:
    """Parsea transcripción cruda de YouTube al formato HH:MM:SS - Texto."""
    print("[3/6] Parseando transcripción de YouTube...")
    
    with open(ruta, 'r', encoding='utf-8') as f:
        contenido = f.read().strip()
    
    lineas = contenido.split('\n')
    segmentos = []
    timestamp_actual = None
    texto_pendiente = ""
    
    RE_TS = re.compile(r'^(\d{1,2}:\d{2}(?::\d{2})?)')
    RE_LIMPIO = re.compile(r'^\d{2}:\d{2}:\d{2}\s*-\s*')
    
    def _ts_a_hhmmss(ts: str) -> str:
        partes = ts.split(':')
        if len(partes) == 2:
            return f"00:{int(partes[0]):02d}:{int(partes[1]):02d}"
        elif len(partes) == 3:
            return f"{int(partes[0]):02d}:{int(partes[1]):02d}:{int(partes[2]):02d}"
        return "00:00:00"
    
    for linea in lineas:
        linea = linea.strip()
        if not linea:
            continue
        
        if RE_LIMPIO.match(linea):
            if texto_pendiente and timestamp_actual:
                segmentos.append((timestamp_actual, texto_pendiente.strip()))
            timestamp_actual = linea[:8]
            texto_pendiente = RE_LIMPIO.sub('', linea).strip()
            continue
        
        match = RE_TS.match(linea)
        if match:
            if texto_pendiente and timestamp_actual:
                segmentos.append((timestamp_actual, texto_pendiente.strip()))
            
            ts_raw = match.group(1)
            timestamp_actual = _ts_a_hhmmss(ts_raw)
            resto = linea[len(ts_raw):]
            resto = _remover_descripcion(resto.strip())
            texto_pendiente = resto
        else:
            if texto_pendiente:
                texto_pendiente += " " + linea
    
    if texto_pendiente and timestamp_actual:
        segmentos.append((timestamp_actual, texto_pendiente.strip()))
    
    with open(salida, 'w', encoding='utf-8') as f:
        for ts, texto in segmentos:
            if texto:
                f.write(f"{ts} - {texto}\n")
    
    print(f"      ✓ {len(segmentos)} segmentos parseados")
    if segmentos:
        print(f"      Rango: {segmentos[0][0]} → {segmentos[-1][0]}")
        print(f"      Verificación - primeros 3 segmentos:")
        for ts, txt in segmentos[:3]:
            print(f"        {ts}: {txt[:70]}")
    
    if segmentos:
        primer_ts = segmentos[0][0]
        primer_texto = segmentos[0][1]
        if primer_ts != "00:00:00":
            print(f"      AVISO: Primer timestamp es {primer_ts}, esperado 00:00:00")
        if primer_texto and primer_texto[0].isdigit():
            print(f"      AVISO: Primer texto empieza con número: '{primer_texto[:30]}'")
    
    print(f"      Guardado: {salida}")
    
    return salida


def transcribir_video(
    video_path: str,
    salida: str = "transcripcion_formatted.txt",
    modelo: str = "large-v3-turbo",
    idioma: str = "es",
) -> str:
    """Transcribe video con faster-whisper y genera transcripcion_formatted.txt."""
    print("[3/6] Transcribiendo con faster-whisper...")
    print(f"      Modelo: {modelo}")
    print(f"      Estimación: ~5-6 min para video de 11min en CPU")
    
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError(
            "faster-whisper no instalado.\n"
            "Instalar: pip install faster-whisper"
        )
    
    device, compute_type = detectar_device()
    print(f"      Device: {device} ({compute_type})")
    
    modelo_mapa = {
        "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
        "turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
        "large-v3": "large-v3",
        "medium": "medium",
        "small": "small",
    }
    model_name = modelo_mapa.get(modelo, "deepdml/faster-whisper-large-v3-turbo-ct2")
    
    audio_temp = _extraer_audio(video_path)
    
    try:
        print(f"      Cargando modelo {model_name}...")
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        
        print(f"      Transcribiendo...")
        segments_gen, info = model.transcribe(
            audio_temp,
            language=idioma,
            word_timestamps=False,
            vad_filter=True,
            beam_size=5,
            temperature=0.0,
        )
        
        duracion_total = info.duration
        segmentos_texto = []
        ultimo_tiempo = 0.0
        
        print(f"      Duración: {duracion_total:.0f}s")
        print(f"      [{'░' * 40}] 0%", end='', flush=True)
        
        inicio = time.time()
        
        for segment in segments_gen:
            segmentos_texto.append({
                "ts": segundos_a_ms(segment.start),
                "texto": segment.text.strip()
            })
            ultimo_tiempo = segment.end
            
            progreso = min(ultimo_tiempo / duracion_total, 1.0)
            pct = int(progreso * 100)
            llenos = int(progreso * 40)
            barra = '█' * llenos + '░' * (40 - llenos)
            
            transcurrido = time.time() - inicio
            velocidad = ultimo_tiempo / max(transcurrido, 0.1)
            eta = int((duracion_total - ultimo_tiempo) / max(velocidad, 0.1))
            
            print(
                f"\r      [{barra}] {pct}% | {velocidad:.1f}x | {eta}s restantes    ",
                end='', flush=True
            )
        
        print(f"\r      [{'█'*40}] 100% ✓                              ")
        
        with open(salida, 'w', encoding='utf-8') as f:
            for seg in segmentos_texto:
                if seg['texto']:
                    f.write(f"{seg['ts']} - {seg['texto']}\n")
        
        print(f"      ✓ {len(segmentos_texto)} segmentos transcritos")
        print(f"      Guardado: {salida}")
        
        return salida
        
    finally:
        try:
            os.remove(audio_temp)
        except:
            pass


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso:")
        print("  Con transcripción: python whisper_transcriber.py transcripcion.txt")
        print("  Con video:         python whisper_transcriber.py video.mp4")
        sys.exit(1)
    
    entrada = sys.argv[1]
    salida = sys.argv[2] if len(sys.argv) > 2 else "transcripcion_formatted.txt"
    
    if entrada.endswith('.txt'):
        parsear_transcripcion_youtube(entrada, salida)
    else:
        transcribir_video(entrada, salida)
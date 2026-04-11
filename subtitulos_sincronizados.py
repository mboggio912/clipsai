#!/usr/bin/env python3
"""
Generador de subtítulos sincronizados palabra por palabra.
"""

import os
import re
import subprocess
from typing import List, Dict


def segundos_a_srt(segundos: float) -> str:
    """Convierte segundos a HH:MM:SS,mmm."""
    h = int(segundos // 3600)
    m = int((segundos % 3600) // 60)
    s = int(segundos % 60)
    ms = int((segundos % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def ts_a_segundos(ts: str) -> float:
    """Convierte HH:MM:SS a segundos."""
    partes = ts.split(':')
    while len(partes) < 3:
        partes.insert(0, '0')
    return int(partes[0]) * 3600 + int(partes[1]) * 60 + float(partes[2])


def parsear_transcripcion(ruta: str) -> List[Dict]:
    """Parsea transcripción con timestamps."""
    segmentos = []
    tiempo = 0.0
    texto = ""
    
    with open(ruta, 'r', encoding='utf-8') as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            
            m = re.match(r'\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?\s*(.+)', linea)
            if m:
                if texto:
                    segmentos.append({'inicio': tiempo, 'fin': tiempo + 3, 'texto': texto})
                    tiempo += 3.5
                try:
                    tiempo = ts_a_segundos(m.group(1))
                except:
                    pass
                texto = m.group(2).strip()
            else:
                texto += " " + linea
    
    if texto:
        segmentos.append({'inicio': tiempo, 'fin': tiempo + 3, 'texto': texto})
    
    return segmentos


def generar_srt(segmentos: List[Dict], salida: str) -> bool:
    """Genera archivo SRT."""
    lineas = []
    n = 1
    
    for seg in segmentos:
        if not seg.get('texto', '').strip():
            continue
        lineas.append(f"{n}")
        lineas.append(f"{segundos_a_srt(seg['inicio'])} --> {segundos_a_srt(seg['fin'])}")
        lineas.append(seg['texto'].strip())
        lineas.append("")
        n += 1
    
    with open(salida, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lineas))
    
    print(f"  SRT: {n-1} subtítulos")
    return True


def quemar_subtitulos(video: str, srt: str, salida: str) -> bool:
    """Quema subtítulos en video con FFmpeg."""
    try:
        filtro = f"subtitles='{srt}',scale={1080}:{1920}"
        
        cmd = [
            "ffmpeg", "-y", "-i", video,
            "-vf", filtro,
            "-c:a", "copy",
            "-crf", "23",
            "-preset", "fast",
            salida
        ]
        
        r = subprocess.run(cmd, capture_output=True, timeout=300)
        
        if r.returncode == 0:
            print(f"  ✓ Subtítulos quemados")
            return True
        else:
            print(f"  Error FFmpeg: {r.stderr.decode()[:100]}")
            return False
            
    except Exception as e:
        print(f"  Error: {e}")
        return False


def crear_subtitulos(video: str, transcripcion: str, salida: str) -> bool:
    """Función principal."""
    print("  Generando subtítulos...")
    
    if not os.path.exists(transcripcion):
        print(f"  No existe: {transcripcion}")
        return False
    
    segmentos = parsear_transcripcion(transcripcion)
    print(f"  Segmentos: {len(segmentos)}")
    
    if not segmentos:
        print("  Sin segmentos")
        return False
    
    srt_path = "temp.srt"
    generar_srt(segmentos, srt_path)
    
    return quemar_subtitulos(video, srt_path, salida)


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 4:
        crear_subtitulos(sys.argv[1], sys.argv[2], sys.argv[3])

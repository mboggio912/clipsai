#!/usr/bin/env python3
"""
Subtítulos sincronizados con animaciones estilo viral TikTok/Reels.
- Cada palabra aparece individualmente
- Alternancia de colores (blanco/amarillo)
- Animación pop-in suave
- Timing basado en timestamps originales de YouTube
"""

import os
import re
import subprocess
from typing import List, Tuple


def ts_a_segundos(ts: str) -> float:
    """Convierte '0:01', '1:01', '1:01:01' a segundos."""
    partes = ts.strip().split(":")
    if len(partes) == 2:
        return int(partes[0]) * 60 + int(partes[1])
    elif len(partes) == 3:
        return int(partes[0]) * 3600 + int(partes[1]) * 60 + int(partes[2])
    return 0.0


def segundos_a_ass(seg: float) -> str:
    """Convierte segundos a H:MM:SS.cc (formato ASS)."""
    h = int(seg // 3600)
    m = int((seg % 3600) // 60)
    s = int(seg % 60)
    cs = int((seg % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def segundos_a_ts(seg: float) -> str:
    """Convierte segundos a HH:MM:SS (para logging)."""
    h = int(seg // 3600)
    m = int((seg % 3600) // 60)
    s = int(seg % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def leer_transcripcion(ruta: str) -> List[Tuple[float, str]]:
    """
    Parsea transcripcion_formatted.txt con formato: HH:MM:SS - Texto
    Returns: Lista de tuplas (tiempo_en_segundos_float, texto_string)
    """
    if not os.path.exists(ruta):
        print(f"  ERROR: No existe {ruta}")
        return []

    with open(ruta, "r", encoding="utf-8") as f:
        lineas = f.readlines()

    segmentos: List[Tuple[float, str]] = []
    patron = re.compile(r'^(\d{2}:\d{2}:\d{2})\s*-\s*(.+)$')

    for linea in lineas:
        linea = linea.strip()
        if not linea:
            continue
        
        match = patron.match(linea)
        if match:
            ts_str = match.group(1)
            texto = match.group(2).strip()
            tiempo = ts_a_segundos(ts_str)
            
            if texto:
                segmentos.append((tiempo, texto))

    print(f"  Transcripción: {len(segmentos)} segmentos leídos de '{ruta}'")
    for t, txt in segmentos[:3]:
        print(f"    {segundos_a_ts(t)}: {txt[:50]}...")
    if len(segmentos) > 3:
        print(f"    ... ({len(segmentos) - 3} más)")

    return segmentos


ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Blanco,Arial,82,&H00FFFFFF,&H000000FF,&H00000000,&H88000000,-1,0,0,0,100,100,2,0,1,4,2,2,60,60,180,1
Style: Amarillo,Arial,82,&H0000FFFF,&H000000FF,&H00000000,&H88000000,-1,0,0,0,100,100,2,0,1,4,2,2,60,60,180,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def generar_ass(
    segmentos: List[Tuple[float, str]],
    ruta_ass: str,
    inicio_clip: float = 0.0,
    duracion_clip: float = None,
) -> bool:
    """
    Genera archivo ASS con subtítulos animados estilo viral.
    
    Características:
    - Cada palabra aparece individualmente
    - Alternancia de colores (blanco/amarillo)
    - Animación pop-in suave
    - Duración por palabra: 0.25s - 1.2s
    """
    if not segmentos:
        print("  AVISO: Lista de segmentos vacía")
        return False
    
    segs_en_rango = []
    for t_abs, texto in segmentos:
        t_rel = t_abs - inicio_clip
        if 0 <= t_rel <= (duracion_clip or float('inf')):
            segs_en_rango.append((t_rel, t_abs, texto))
    
    total_segmentos = len(segmentos)
    en_rango = len(segs_en_rango)
    
    print(f"  Total segmentos en transcripción: {total_segmentos}")
    print(f"  Segmentos dentro del clip: {en_rango}")
    
    if en_rango == 0:
        print(f"  AVISO: Ningún segmento cae en el rango del clip")
        return False
    
    lineas_ass = [ASS_HEADER]
    total_lineas_ass = 0
    palabra_idx = 0
    
    for i, (t_rel, t_abs, texto) in enumerate(segs_en_rango):
        t_siguiente_rel = segs_en_rango[i + 1][0] if i + 1 < len(segs_en_rango) else t_rel + 3.0
        duracion_segmento = t_siguiente_rel - t_rel
        
        palabras = texto.split()
        num_palabras = len(palabras)
        
        if num_palabras == 0:
            continue
        
        dur_por_palabra = duracion_segmento / num_palabras
        dur_por_palabra = max(0.25, min(1.2, dur_por_palabra))
        
        for j, palabra in enumerate(palabras):
            p_ini = t_rel + j * dur_por_palabra
            p_fin = p_ini + dur_por_palabra
            
            estilo = "Blanco" if palabra_idx % 2 == 0 else "Amarillo"
            palabra_idx += 1
            
            animacion = r"{\fad(60,100)\fscx115\fscy115\t(0,120,\fscx108\fscy108)}"
            texto_ass = palabra.upper()
            
            ini_ass = segundos_a_ass(p_ini)
            fin_ass = segundos_a_ass(p_fin)
            
            linea = f"Dialogue: 0,{ini_ass},{fin_ass},{estilo},,0,0,0,,{animacion}{texto_ass}"
            lineas_ass.append(linea)
            total_lineas_ass += 1
    
    with open(ruta_ass, "w", encoding="utf-8", newline='\n') as f:
        f.write('\n'.join(lineas_ass) + '\n')
    
    print(f"  Líneas ASS generadas: {total_lineas_ass} (palabras)")
    
    return True


def crear_video_con_subtitulos(
    video_entrada: str,
    segmentos: List[Tuple[float, str]],
    video_salida: str,
    inicio_clip: float = 0.0,
    duracion_clip: float = None,
) -> bool:
    """
    Genera el ASS y lo quema en el video con FFmpeg.
    """
    if not segmentos:
        print("  ERROR: Sin segmentos para generar subtítulos")
        return False

    ruta_ass = video_salida.replace(".mp4", "_subs.ass")

    print(f"  Generando subtítulos virales animados...")
    ok = generar_ass(segmentos, ruta_ass, inicio_clip, duracion_clip)
    
    if not ok:
        print("  Subtítulos omitidos (sin segmentos en rango)")
        return False

    ruta_ass_escaped = ruta_ass.replace("\\", "/").replace(":", "\\:")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_entrada,
        "-vf", f"ass='{ruta_ass_escaped}'",
        "-c:a", "copy",
        "-preset", "ultrafast",
        "-threads", "4",
        video_salida,
    ]

    print(f"  Quemando subtítulos en video...")
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=300)
        if r.returncode == 0:
            print("  ✓ Subtítulos virales aplicados")
            try:
                os.remove(ruta_ass)
            except Exception:
                pass
            return True
        else:
            err = r.stderr.decode(errors="replace")[:500]
            print(f"  Error FFmpeg:\n{err}")
            return False
    except subprocess.TimeoutExpired:
        print("  Error: FFmpeg tardó demasiado")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Uso: python sub_reales.py video.mp4 transcripcion.txt salida.mp4 [inicio_seg]")
        print("  inicio_seg: segundo de inicio del clip dentro del video original (default 0)")
        sys.exit(1)

    video = sys.argv[1]
    trans = sys.argv[2]
    salida = sys.argv[3]
    inicio = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0

    segmentos = leer_transcripcion(trans)
    print(f"\nTotal: {len(segmentos)} segmentos")

    if not segmentos:
        print("No se encontraron segmentos")
        sys.exit(1)

    crear_video_con_subtitulos(video, segmentos, salida, inicio_clip=inicio)

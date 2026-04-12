#!/usr/bin/env python3
"""
Subtítulos sincronizados con timing preciso basado en timestamps de YouTube.
Los timestamps son la fuente de verdad - no se modifican.
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
    
    Returns:
        Lista de tuplas (tiempo_en_segundos_float, texto_string)
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
        else:
            print(f"  AVISO: Línea ignorada (formato inválido): {linea[:50]}...")

    print(f"  Transcripción: {len(segmentos)} segmentos leídos de '{ruta}'")
    for t, txt in segmentos[:3]:
        print(f"    {segundos_a_ts(t)}: {txt[:50]}...")
    if len(segmentos) > 3:
        print(f"    ... ({len(segmentos) - 3} más)")

    return segmentos


def _dividir_texto(texto: str, palabras_por_chunk: int = 6) -> List[str]:
    """Divide texto en chunks de palabras."""
    palabras = texto.split()
    if len(palabras) <= palabras_por_chunk:
        return [texto]
    
    chunks = []
    for i in range(0, len(palabras), palabras_por_chunk):
        chunk = ' '.join(palabras[i:i + palabras_por_chunk])
        chunks.append(chunk)
    
    return chunks


MAX_DURACION_SEGMENTO = 2.5


def generar_ass(
    segmentos: List[Tuple[float, str]],
    ruta_ass: str,
    inicio_clip: float = 0.0,
    duracion_clip: float = None,
) -> bool:
    """
    Genera archivo ASS con subtítulos sincronizados.
    
    Lógica de timing:
    - Cada segmento tiene timestamp ABSOLUTO del video original
    - Convertimos a tiempo RELATIVO restando inicio_clip
    - Filtramos solo segmentos dentro del rango del clip
    - Si un segmento dura más de MAX_DURACION_SEGMENTO (2.5s), lo dividimos en chunks
    """
    if not segmentos:
        print("  AVISO: Lista de segmentos vacía")
        return False
    
    fin_clip = inicio_clip + duracion_clip if duracion_clip else float('inf')
    
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
        print(f"  Clip: {segundos_a_ts(inicio_clip)} -> {segundos_a_ts(fin_clip)}")
        return False
    
    lineas_ass = []
    lineas_ass.append("[Script Info]")
    lineas_ass.append("ScriptType: v4.00+")
    lineas_ass.append("PlayResX: 1080")
    lineas_ass.append("PlayResY: 1920")
    lineas_ass.append("ScaledBorderAndShadow: yes")
    lineas_ass.append("")
    lineas_ass.append("[V4+ Styles]")
    lineas_ass.append("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding")
    lineas_ass.append("Style: Viral,Arial,78,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,2,0,1,5,2,2,60,60,140,1")
    lineas_ass.append("")
    lineas_ass.append("[Events]")
    lineas_ass.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")
    
    total_lineas_ass = 0
    primer_tiempo = None
    ultimo_tiempo = None
    
    for i, (t_rel, t_abs, texto) in enumerate(segs_en_rango):
        t_siguiente_rel = segs_en_rango[i + 1][0] if i + 1 < len(segs_en_rango) else t_rel + 3.0
        duracion_natural = t_siguiente_rel - t_rel
        
        if duracion_natural > MAX_DURACION_SEGMENTO and len(texto.split()) > 6:
            n_chunks = max(2, int(duracion_natural / MAX_DURACION_SEGMENTO) + 1)
            chunks = _dividir_texto(texto, palabras_por_chunk=6)
            n_chunks = min(n_chunks, len(chunks))
            
            dur_por_chunk = duracion_natural / n_chunks
            
            for j in range(n_chunks):
                chunk_texto = chunks[j] if j < len(chunks) else chunks[0]
                c_ini = t_rel + j * dur_por_chunk
                c_fin = c_ini + dur_por_chunk - 0.05
                
                if primer_tiempo is None:
                    primer_tiempo = c_ini
                ultimo_tiempo = c_fin
                
                ini_ass = segundos_a_ass(c_ini)
                fin_ass = segundos_a_ass(c_fin)
                
                linea = f"Dialogue: 0,{ini_ass},{fin_ass},Viral,,0,0,0,,{{\\fad(80,150)}}{chunk_texto.upper()}"
                lineas_ass.append(linea)
                total_lineas_ass += 1
        else:
            t_ini = t_rel
            t_fin = min(t_siguiente_rel - 0.05, t_rel + MAX_DURACION_SEGMENTO)
            
            if primer_tiempo is None:
                primer_tiempo = t_ini
            ultimo_tiempo = t_fin
            
            ini_ass = segundos_a_ass(t_ini)
            fin_ass = segundos_a_ass(t_fin)
            
            linea = f"Dialogue: 0,{ini_ass},{fin_ass},Viral,,0,0,0,,{{\\fad(80,150)}}{texto.upper()}"
            lineas_ass.append(linea)
            total_lineas_ass += 1
    
    with open(ruta_ass, "w", encoding="utf-8") as f:
        f.write('\n'.join(lineas_ass) + '\n')
    
    print(f"  Líneas ASS generadas: {total_lineas_ass}")
    if primer_tiempo is not None and ultimo_tiempo is not None:
        print(f"  Rango de tiempo: {segundos_a_ts(primer_tiempo)} -> {segundos_a_ts(ultimo_tiempo)}")
    
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

    print(f"  Generando subtítulos sincronizados...")
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
            print("  ✓ Subtítulos sincronizados aplicados")
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

#!/usr/bin/env python3
"""
Sistema profesional de subtítulos para contenido viral.
Modo A: Whisper (whisper_words.json) - sincronización perfecta por palabra
Modo B: Fallback (transcripcion_formatted.txt) - cuando no hay Whisper

Animaciones estilo TikTok/Reels profesional:
- Palabra por palabra con timing exacto
- Alternancia blanco/amarillo
- Pop-in con escala
- Highlight de palabra activa en color
- Sombra para legibilidad sobre cualquier fondo
"""

import os
import json
import re
import subprocess
from typing import List, Tuple, Dict, Optional


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


def ts_a_segundos(ts: str) -> float:
    """Convierte '0:01', '1:01', '1:01:01' a segundos."""
    partes = ts.strip().split(":")
    if len(partes) == 2:
        return int(partes[0]) * 60 + int(partes[1])
    elif len(partes) == 3:
        return int(partes[0]) * 3600 + int(partes[1]) * 60 + int(partes[2])
    return 0.0


def crear_subtitulos_clip(
    video_entrada: str,
    video_salida: str,
    inicio_clip: float = 0.0,
    duracion_clip: float = None,
    words_json: str = "whisper_words.json",
    transcripcion_fallback: str = "transcripcion_formatted.txt",
) -> bool:
    """
    Detecta automáticamente si usar Modo A o Modo B.

    Modo A (Whisper): lee whisper_words.json, filtra por rango del clip,
                      genera ASS con timing exacto por palabra
    Modo B (Fallback): lee transcripcion_formatted.txt, usa lógica anterior
    """
    if os.path.exists(words_json):
        print(f"  Modo A: Whisper words ({words_json})")
        return _subtitulos_modo_whisper(
            video_entrada, video_salida,
            words_json, inicio_clip, duracion_clip
        )
    elif os.path.exists(transcripcion_fallback):
        print(f"  Modo B: Fallback ({transcripcion_fallback})")
        return _subtitulos_modo_fallback(
            video_entrada, video_salida,
            transcripcion_fallback, inicio_clip, duracion_clip
        )
    else:
        print("  Sin fuente de subtítulos disponible")
        return False


def _subtitulos_modo_whisper(
    video_entrada: str,
    video_salida: str,
    words_json: str,
    inicio_clip: float,
    duracion_clip: float,
) -> bool:
    with open(words_json, 'r', encoding='utf-8') as f:
        words = json.load(f)

    fin_clip = inicio_clip + (duracion_clip or float('inf'))

    words_clip: List[Dict] = []
    for w in words:
        if inicio_clip <= w['start'] < fin_clip:
            words_clip.append({
                "word": w['word'],
                "start": round(w['start'] - inicio_clip, 3),
                "end": round(min(w['end'], fin_clip) - inicio_clip, 3),
                "probability": w.get('probability', 1.0)
            })

    print(f"  Palabras totales en JSON: {len(words)}")
    print(f"  Palabras en rango del clip: {len(words_clip)}")

    if not words_clip:
        print("  AVISO: Sin palabras en el rango del clip")
        return False

    ruta_ass = video_salida.replace('.mp4', '_pro.ass')
    ok = _generar_ass_whisper(words_clip, ruta_ass)

    if not ok:
        return False

    return _quemar_ass(video_entrada, ruta_ass, video_salida)


def _generar_ass_whisper(words_clip: list, ruta_ass: str) -> bool:
    ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Blanco,Arial,88,&H00FFFFFF,&H000000FF,&H00000000,&H99000000,-1,0,0,0,100,100,2,0,1,4,3,2,60,60,200,1
Style: Amarillo,Arial,88,&H0000FFFF,&H000000FF,&H00000000,&H99000000,-1,0,0,0,100,100,2,0,1,4,3,2,60,60,200,1
Style: Highlight,Arial,94,&H000099FF,&H000000FF,&H00000000,&HCC000000,-1,0,0,0,100,100,2,0,1,5,3,2,60,60,200,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lineas = [ASS_HEADER]
    palabra_idx = 0

    for w in words_clip:
        palabra = w['word'].strip()
        if not palabra:
            continue

        t_ini = w['start']
        t_fin = w['end']

        if t_fin - t_ini < 0.15:
            t_fin = t_ini + 0.15

        ini_ass = segundos_a_ass(t_ini)
        fin_ass = segundos_a_ass(t_fin)

        if palabra_idx % 8 == 7:
            estilo = "Highlight"
            efecto = (
                r"{\fad(40,80)"
                r"\fscx120\fscy120"
                r"\t(0,100,\fscx105\fscy105)"
                r"\blur1}"
            )
        elif palabra_idx % 2 == 0:
            estilo = "Blanco"
            efecto = (
                r"{\fad(50,80)"
                r"\fscx112\fscy112"
                r"\t(0,100,\fscx103\fscy103)}"
            )
        else:
            estilo = "Amarillo"
            efecto = (
                r"{\fad(50,80)"
                r"\fscx112\fscy112"
                r"\t(0,100,\fscx103\fscy103)}"
            )

        texto_ass = palabra.upper()
        linea = f"Dialogue: 0,{ini_ass},{fin_ass},{estilo},,0,0,0,,{efecto}{texto_ass}"
        lineas.append(linea)
        palabra_idx += 1

    with open(ruta_ass, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lineas) + '\n')

    print(f"  ASS generado: {len(lineas) - 1} palabras con timing exacto")
    return True


def _subtitulos_modo_fallback(
    video_entrada: str,
    video_salida: str,
    transcripcion_fallback: str,
    inicio_clip: float,
    duracion_clip: float,
) -> bool:
    """Fallback: usa transcripcion_formatted.txt con timing aproximado."""
    if not os.path.exists(transcripcion_fallback):
        print(f"  ERROR: No existe {transcripcion_fallback}")
        return False

    with open(transcripcion_fallback, 'r', encoding='utf-8') as f:
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

    if not segmentos:
        print("  ERROR: Sin segmentos en transcripción")
        return False

    segs_en_rango: List[Tuple[float, float, str]] = []
    fin_clip = inicio_clip + (duracion_clip or float('inf'))
    for t_abs, texto in segmentos:
        t_rel = t_abs - inicio_clip
        if 0 <= t_rel <= (duracion_clip or float('inf')):
            segs_en_rango.append((t_rel, t_abs, texto))

    if not segs_en_rango:
        print("  AVISO: Ningún segmento cae en el rango del clip")
        return False

    ruta_ass = video_salida.replace('.mp4', '_pro.ass')
    ok = _generar_ass_fallback(segs_en_rango, ruta_ass)

    if not ok:
        return False

    return _quemar_ass(video_entrada, ruta_ass, video_salida)


def _generar_ass_fallback(segs_en_rango: List[Tuple[float, float, str]], ruta_ass: str) -> bool:
    """Genera ASS desde segmentos (fallback)."""
    ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Blanco,Arial,88,&H00FFFFFF,&H000000FF,&H00000000,&H99000000,-1,0,0,0,100,100,2,0,1,4,3,2,60,60,200,1
Style: Amarillo,Arial,88,&H0000FFFF,&H000000FF,&H00000000,&H99000000,-1,0,0,0,100,100,2,0,1,4,3,2,60,60,200,1
Style: Highlight,Arial,94,&H000099FF,&H000000FF,&H00000000,&HCC000000,-1,0,0,0,100,100,2,0,1,5,3,2,60,60,200,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lineas = [ASS_HEADER]
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

            if p_fin - p_ini < 0.15:
                p_fin = p_ini + 0.15

            ini_ass = segundos_a_ass(p_ini)
            fin_ass = segundos_a_ass(p_fin)

            if palabra_idx % 8 == 7:
                estilo = "Highlight"
                efecto = r"{\fad(40,80)\fscx120\fscy120\t(0,100,\fscx105\fscy105)\blur1}"
            elif palabra_idx % 2 == 0:
                estilo = "Blanco"
                efecto = r"{\fad(50,80)\fscx112\fscy112\t(0,100,\fscx103\fscy103)}"
            else:
                estilo = "Amarillo"
                efecto = r"{\fad(50,80)\fscx112\fscy112\t(0,100,\fscx103\fscy103)}"

            texto_ass = palabra.upper()
            linea = f"Dialogue: 0,{ini_ass},{fin_ass},{estilo},,0,0,0,,{efecto}{texto_ass}"
            lineas.append(linea)
            palabra_idx += 1

    with open(ruta_ass, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lineas) + '\n')

    print(f"  ASS generado (fallback): {len(lineas) - 1} palabras")
    return True


def _quemar_ass(video_entrada: str, ruta_ass: str, video_salida: str) -> bool:
    """Quema ASS en video con FFmpeg."""
    ruta_escaped = ruta_ass.replace('\\', '/')
    if len(ruta_escaped) > 1 and ruta_escaped[1] == ':':
        ruta_escaped = ruta_escaped[0] + '\\:' + ruta_escaped[2:]

    cmd = [
        "ffmpeg", "-y",
        "-i", video_entrada,
        "-vf", f"ass='{ruta_escaped}'",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "17",
        "-c:a", "copy",
        "-threads", "4",
        video_salida
    ]

    try:
        r = subprocess.run(cmd, capture_output=True, timeout=300)
        if r.returncode == 0:
            print(f"  ✓ Subtítulos quemados en video")
            try:
                os.remove(ruta_ass)
            except Exception:
                pass
            return True
        else:
            err = r.stderr.decode(errors='replace')[:400]
            print(f"  Error FFmpeg:\n{err}")
            return False
    except subprocess.TimeoutExpired:
        print("  Error: timeout en FFmpeg")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Uso: python subtitulos_pro.py video.mp4 salida.mp4 inicio_seg [duracion_seg]")
        print("  Modo A: usa whisper_words.json (sincronización perfecta)")
        print("  Modo B: usa transcripcion_formatted.txt (fallback)")
        sys.exit(1)

    video = sys.argv[1]
    salida = sys.argv[2]
    inicio = float(sys.argv[3])
    duracion = float(sys.argv[4]) if len(sys.argv) > 4 else None

    ok = crear_subtitulos_clip(video, salida, inicio, duracion)
    if ok:
        print(f"\n✓ Video con subtítulos: {salida}")
    else:
        print("\nError al generar subtítulos")
        sys.exit(1)

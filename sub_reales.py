#!/usr/bin/env python3
"""
Subtítulos animados estilo viral generados desde transcripcion.txt.
Usa formato ASS para animaciones reales (fade + pop-in) quemadas con FFmpeg.
"""
 
import os
import re
import subprocess
from typing import List, Tuple
 
 
# ─────────────────────────────────────────────
#  PARSEO DEL ARCHIVO DE TRANSCRIPCIÓN
# ─────────────────────────────────────────────
 
def ts_a_segundos(ts: str) -> float:
    """Convierte '0:01', '1:01', '1:01:01' a segundos."""
    partes = ts.strip().split(":")
    if len(partes) == 2:
        return int(partes[0]) * 60 + int(partes[1])
    elif len(partes) == 3:
        return int(partes[0]) * 3600 + int(partes[1]) * 60 + int(partes[2])
    return 0.0
 
 
def segundos_a_ts(seg: float) -> str:
    """Convierte segundos a HH:MM:SS (para logging)."""
    h = int(seg // 3600)
    m = int((seg % 3600) // 60)
    s = int(seg % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
 
 
def leer_transcripcion(ruta: str) -> List[Tuple[float, str]]:
    """
    Parsea transcripcion.txt con el formato:
        0:01
        1 segundo          ← línea redundante, se ignora
        Texto del segmento
 
    Devuelve lista de (tiempo_en_segundos, texto).
    """
    if not os.path.exists(ruta):
        print(f"  ERROR: No existe {ruta}")
        return []
 
    with open(ruta, "r", encoding="utf-8") as f:
        lineas = [l.rstrip() for l in f.readlines()]
 
    # Patrones de reconocimiento
    RE_TS       = re.compile(r"^(\d{1,2}:\d{2}(?::\d{2})?)$")
    RE_SEG      = re.compile(r"^\d+\s+segundos?$", re.IGNORECASE)
    RE_MIN      = re.compile(r"^\d+\s+minutos?(?:\s+y\s+\d+\s+segundos?)?$", re.IGNORECASE)
    RE_REDUNDANTE = re.compile(r"^\d+\s+(segundo|minuto)", re.IGNORECASE)
 
    segmentos: List[Tuple[float, str]] = []
    tiempo_actual = 0.0
    texto_pendiente = ""
 
    def guardar():
        nonlocal texto_pendiente
        t = texto_pendiente.strip()
        if t:
            segmentos.append((tiempo_actual, t))
        texto_pendiente = ""
 
    for linea in lineas:
        if not linea.strip():
            continue
 
        m_ts = RE_TS.match(linea.strip())
        if m_ts:
            # Nueva marca de tiempo → guardar texto anterior primero
            guardar()
            tiempo_actual = ts_a_segundos(m_ts.group(1))
            continue
 
        # Ignorar líneas redundantes "X segundos", "X minutos y Y segundos"
        if RE_REDUNDANTE.match(linea.strip()):
            continue
 
        # Línea de texto
        if texto_pendiente:
            texto_pendiente += " " + linea.strip()
        else:
            texto_pendiente = linea.strip()
 
    guardar()  # último segmento
 
    print(f"  Transcripción: {len(segmentos)} segmentos leídos de '{ruta}'")
    for t, txt in segmentos[:5]:
        print(f"    {segundos_a_ts(t)}: {txt[:60]}")
    if len(segmentos) > 5:
        print(f"    ... ({len(segmentos) - 5} más)")
 
    return segmentos
 
 
# ─────────────────────────────────────────────
#  GENERACIÓN DE ARCHIVO ASS ANIMADO
# ─────────────────────────────────────────────
 
ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes
 
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Viral,Arial,80,&H00FFFFFF,&H000000FF,&H00000000,&HAA000000,-1,0,0,0,100,100,2,0,1,4,2,2,60,60,120,1
 
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
 
 
def segundos_a_ass(seg: float) -> str:
    """Convierte segundos a H:MM:SS.cc (formato ASS)."""
    h = int(seg // 3600)
    m = int((seg % 3600) // 60)
    s = int(seg % 60)
    cs = int((seg % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _dividir_en_palabras(texto: str) -> List[str]:
    """Divide texto en palabras individuales."""
    return texto.split()


def generar_ass(
    segmentos: List[Tuple[float, str]],
    ruta_ass: str,
    inicio_clip: float = 0.0,
    duracion_clip: float = None,
) -> bool:
    """
    Genera archivo ASS con animación palabra por palabra (estilo karaoke).
    """
    lineas_ass = []
    idx = 1
    
    seg_clip = []
    for tiempo, texto in segmentos:
        t_rel = tiempo - inicio_clip
        if t_rel < 0:
            continue
        if duracion_clip is not None and t_rel > duracion_clip:
            break
        seg_clip.append((t_rel, texto))
    
    if not seg_clip:
        print("  AVISO: Ningún segmento cae dentro del rango del clip.")
        return False
    
    for i, (t_inicio, texto) in enumerate(seg_clip):
        t_fin = seg_clip[i + 1][0] - 0.05 if i + 1 < len(seg_clip) else t_inicio + 4.0
        if duracion_clip is not None:
            t_fin = min(t_fin, duracion_clip)
        
        palabras = _dividir_en_palabras(texto)
        if not palabras:
            continue
        
        duracion_segmento = t_fin - t_inicio
        duracion_palabra = duracion_segmento / len(palabras)
        
        for j, palabra in enumerate(palabras):
            p_ini = t_inicio + j * duracion_palabra
            p_fin = p_ini + duracion_palabra * 0.85
            
            ini_ass = segundos_a_ass(p_ini)
            fin_ass = segundos_a_ass(p_fin)
            
            efecto = r"{\fad(50,100)}"
            texto_esc = palabra.upper()
            
            linea = (
                f"Dialogue: 0,{ini_ass},{fin_ass},Viral,,0,0,0,,"
                f"{efecto}{texto_esc}"
            )
            lineas_ass.append(linea)
            idx += 1
    
    with open(ruta_ass, "w", encoding="utf-8") as f:
        f.write(ASS_HEADER)
        f.write("\n".join(lineas_ass) + "\n")
    
    print(f"  ASS generado: {ruta_ass} ({idx-1} palabras)")
    return True
 
 
# ─────────────────────────────────────────────
#  QUEMAR SUBTÍTULOS EN VIDEO
# ─────────────────────────────────────────────
 
def crear_video_con_subtitulos(
    video_entrada: str,
    segmentos: List[Tuple[float, str]],
    video_salida: str,
    inicio_clip: float = 0.0,
    duracion_clip: float = None,
) -> bool:
    """
    Genera el ASS y lo quema en el video con FFmpeg.
    Los subtítulos son animados (pop-in + fade) estilo TikTok/Reels.
    """
    if not segmentos:
        print("  ERROR: Sin segmentos para generar subtítulos.")
        return False
 
    ruta_ass = video_salida.replace(".mp4", "_subs.ass")
 
    ok = generar_ass(segmentos, ruta_ass, inicio_clip, duracion_clip)
    if not ok:
        print("  Subtítulos omitidos (sin segmentos en rango).")
        # Copiar el video sin subtítulos como fallback
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_entrada, "-c", "copy", video_salida],
            capture_output=True,
        )
        return False
 
    # FFmpeg quema el ASS directamente (no necesita libass separado)
    ruta_ass_escaped = ruta_ass.replace("\\", "/").replace(":", "\\:")
 
    cmd = [
        "ffmpeg", "-y",
        "-i", video_entrada,
        "-vf", f"ass='{ruta_ass_escaped}'",
        "-c:a", "copy",
        "-preset", "fast",
        video_salida,
    ]
 
    print(f"  Quemando subtítulos animados → {os.path.basename(video_salida)}")
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=300)
        if r.returncode == 0:
            print("  ✓ Subtítulos animados aplicados correctamente.")
            # Limpiar temporal
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
        print("  Error: FFmpeg tardó demasiado.")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False
 
 
# ─────────────────────────────────────────────
#  USO DIRECTO
# ─────────────────────────────────────────────
 
def main():
    import sys
 
    if len(sys.argv) < 4:
        print("Uso: python sub_reales.py video.mp4 transcripcion.txt salida.mp4 [inicio_seg]")
        print("  inicio_seg: segundo de inicio del clip dentro del video original (default 0)")
        return
 
    video   = sys.argv[1]
    trans   = sys.argv[2]
    salida  = sys.argv[3]
    inicio  = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
 
    segmentos = leer_transcripcion(trans)
    print(f"\nTotal: {len(segmentos)} segmentos")
 
    if not segmentos:
        print("No se encontraron segmentos. Revisá el formato del archivo.")
        return
 
    crear_video_con_subtitulos(video, segmentos, salida, inicio_clip=inicio)
 
 
if __name__ == "__main__":
    main()
 
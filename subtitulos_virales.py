#!/usr/bin/env python3
"""
Generador de subtítulos animados estilo viral (TikTok / Shorts / Reels).
Usa formato ASS con efectos reales: pop-in, color dinámico, shake en momentos intensos.
"""
 
import os
import subprocess
from typing import List, Dict, Optional
 
 
# ─────────────────────────────────────────────
#  HELPERS DE TIEMPO
# ─────────────────────────────────────────────
 
def segundos_a_ass(seg: float) -> str:
    """Segundos → H:MM:SS.cc (formato ASS)."""
    h  = int(seg // 3600)
    m  = int((seg % 3600) // 60)
    s  = int(seg % 60)
    cs = int((seg % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
 
 
# ─────────────────────────────────────────────
#  CABECERA ASS CON ESTILO VIRAL
# ─────────────────────────────────────────────
 
def _cabecera_ass(ancho: int = 1080, alto: int = 1920) -> str:
    """Genera cabecera ASS con estilo TikTok/Reels."""
    return f"""\
[Script Info]
ScriptType: v4.00+
PlayResX: {ancho}
PlayResY: {alto}
ScaledBorderAndShadow: yes
 
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Viral,Arial,82,&H00FFFFFF,&H000000FF,&H00000000,&HAA000000,-1,0,0,0,100,100,3,0,1,5,3,2,60,60,140,1
Style: Hook,Arial,96,&H0000FFFF,&H000000FF,&H00000000,&HAA000000,-1,0,0,0,100,100,2,0,1,6,3,5,80,80,{alto//2 - 80},1
Style: Intenso,Arial,90,&H000099FF,&H000000FF,&H00000000,&HAA000000,-1,0,0,0,100,100,3,0,1,5,3,2,60,60,140,1
 
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
 
 
# ─────────────────────────────────────────────
#  CONSTRUCCIÓN DE LÍNEAS ASS
# ─────────────────────────────────────────────
 
def _linea_ass(inicio: float, fin: float, texto: str, estilo: str = "Viral",
               intenso: bool = False) -> str:
    """
    Genera una línea Dialogue ASS con:
      - Fade in 120 ms / fade out 180 ms
      - Pop-in: escala 115→100 en 200 ms
      - Si intenso: color naranja + escala extra 125→100
    """
    ini = segundos_a_ass(inicio)
    fin_ass = segundos_a_ass(fin)
 
    if intenso:
        # Naranja brillante + pop más agresivo
        efecto = (
            r"{\fad(80,180)"
            r"\c&H0033CCFF&"          # naranja (BGR en ASS)
            r"\fscx125\fscy125"
            r"\t(0,200,\fscx100\fscy100\c&H00FFFFFF&)}"
        )
    else:
        efecto = (
            r"{\fad(120,180)"
            r"\fscx115\fscy115"
            r"\t(0,200,\fscx100\fscy100)}"
        )
 
    return f"Dialogue: 0,{ini},{fin_ass},{estilo},,0,0,0,,{efecto}{texto.upper()}"
 
 
def _linea_hook(inicio: float, fin: float, texto: str) -> str:
    """Hook inicial en el centro de la pantalla, amarillo, más grande."""
    ini    = segundos_a_ass(inicio)
    fin_ass = segundos_a_ass(fin)
    efecto = (
        r"{\fad(100,300)"
        r"\fscx130\fscy130"
        r"\t(0,300,\fscx100\fscy100)"
        r"\c&H00FFFF00&}"           # amarillo (BGR)
    )
    return f"Dialogue: 0,{ini},{fin_ass},Hook,,0,0,0,,{efecto}{texto.upper()}"
 
 
# ─────────────────────────────────────────────
#  DIVISIÓN INTELIGENTE DE TEXTO
# ─────────────────────────────────────────────
 
def _bloques(texto: str, max_palabras: int = 5) -> List[str]:
    """Divide texto en bloques cortos para pantalla vertical."""
    palabras = texto.split()
    resultado = []
    for i in range(0, len(palabras), max_palabras):
        resultado.append(" ".join(palabras[i:i + max_palabras]))
    return resultado or [texto]
 
 
# ─────────────────────────────────────────────
#  API PRINCIPAL
# ─────────────────────────────────────────────
 
def aplicar_subtitulos_virales(
    video_path: str,
    subtitulos: List[Dict],
    salida: str,
    momentos_intensos: Optional[List[Dict]] = None,
    hook_text: Optional[str] = None,
) -> bool:
    """
    Aplica subtítulos animados estilo viral al video.
 
    Args:
        video_path:       Ruta al video de entrada.
        subtitulos:       Lista de dicts {'inicio': float, 'fin': float, 'texto': str}
        salida:           Ruta del video de salida.
        momentos_intensos: Lista de dicts {'timestamp': float, 'intensidad': float}
                           para resaltar en naranja.
        hook_text:        Texto de hook que aparece los primeros 3 s (None = sin hook).
    """
    ruta_ass = salida.replace(".mp4", "_viral.ass")
    ok = _generar_ass_viral(
        subtitulos, ruta_ass, momentos_intensos, hook_text
    )
    if not ok:
        return False
    return _quemar_ass(video_path, ruta_ass, salida)
 
 
def crear_clip_con_efectos(
    video_path: str,
    salida: str,
    subtitulos: Optional[List[Dict]] = None,
    momentos: Optional[List[Dict]] = None,
    hook_text: Optional[str] = None,
) -> bool:
    """
    Versión compatible con editor_viral.py.
    Crea un clip con subtítulos animados y hook inicial.
    """
    subs = subtitulos or []
    return aplicar_subtitulos_virales(
        video_path, subs, salida,
        momentos_intensos=momentos,
        hook_text=hook_text,
    )
 
 
# ─────────────────────────────────────────────
#  GENERACIÓN DEL ARCHIVO ASS
# ─────────────────────────────────────────────
 
def _generar_ass_viral(
    subtitulos: List[Dict],
    ruta_ass: str,
    momentos_intensos: Optional[List[Dict]],
    hook_text: Optional[str],
) -> bool:
    """Escribe el archivo .ass con todos los efectos."""
 
    # Índice de momentos intensos por segundo (para lookup rápido)
    intensos_set: set = set()
    if momentos_intensos:
        for m in momentos_intensos:
            ts = m.get("timestamp", 0)
            if m.get("intensidad", 0) >= 7:
                intensos_set.add(int(ts))
 
    lineas: List[str] = [_cabecera_ass()]
 
    # Hook inicial
    if hook_text:
        lineas.append(_linea_hook(0.0, 3.0, hook_text))
 
    # Subtítulos
    for sub in subtitulos:
        inicio = float(sub.get("inicio", 0))
        fin    = float(sub.get("fin", inicio + 3))
        texto  = sub.get("texto", "").strip()
        if not texto:
            continue
 
        bloques = _bloques(texto, max_palabras=5)
        dur_bloque = (fin - inicio) / len(bloques)
 
        for j, bloque in enumerate(bloques):
            b_ini = inicio + j * dur_bloque
            b_fin = b_ini + dur_bloque
            intenso = int(b_ini) in intensos_set
 
            lineas.append(_linea_ass(b_ini, b_fin, bloque, intenso=intenso))
 
    with open(ruta_ass, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")
 
    print(f"  ASS viral generado: {ruta_ass} ({len(lineas)-1} líneas)")
    return True
 
 
def _quemar_ass(video_path: str, ruta_ass: str, salida: str) -> bool:
    """Quema el archivo ASS en el video con FFmpeg."""
    ass_escaped = ruta_ass.replace("\\", "/")
    if ":" in ass_escaped and not ass_escaped.startswith("/"):
        ass_escaped = ass_escaped.replace(":", "\\:", 1)

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"ass='{ass_escaped}'",
        "-c:a", "copy",
        "-preset", "ultrafast",
        "-threads", "4",
        salida,
    ]

    print(f"  Quemando subtítulos virales → {os.path.basename(salida)}")
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=300)
        if r.returncode == 0:
            print(f"  ✓ Subtítulos virales aplicados: {os.path.basename(salida)}")
            try:
                os.remove(ruta_ass)
            except Exception:
                pass
            return True
        else:
            err = r.stderr.decode(errors="replace")[:600]
            print(f"  Error FFmpeg:\n{err}")
            return False
    except subprocess.TimeoutExpired:
        print("  Error: tiempo de espera agotado en FFmpeg.")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False
 
 
# ─────────────────────────────────────────────
#  USO DIRECTO
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    import sys
 
    if len(sys.argv) < 3:
        print("Uso: python subtitulos_virales.py video.mp4 salida.mp4")
        sys.exit(1)
 
    video_path = sys.argv[1]
    salida     = sys.argv[2]
 
    # Ejemplo demo
    subs_demo = [
        {"inicio": 0,  "fin": 3,  "texto": "Mirá esto no puede ser"},
        {"inicio": 3,  "fin": 6,  "texto": "Lo que pasó después te va a volar la cabeza"},
        {"inicio": 6,  "fin": 10, "texto": "Nunca vi algo así en mi vida"},
        {"inicio": 10, "fin": 14, "texto": "Papi no me lo puedo creer"},
    ]
    momentos_demo = [
        {"timestamp": 3.5, "intensidad": 9},
        {"timestamp": 10,  "intensidad": 8},
    ]
 
    crear_clip_con_efectos(
        video_path, salida,
        subtitulos=subs_demo,
        momentos=momentos_demo,
        hook_text="MIRÁ ESTO 🤯",
    )
 
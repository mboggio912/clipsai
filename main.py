#!/usr/bin/env python3
"""
Generador automático de clips virales.
Extracción de audio + análisis automático + IA + recorte.
"""

import os
import re
import json
import subprocess
import sys
import requests
from typing import List, Dict, Optional

try:
    from audio_analyzer import analizar_audio, guardar_json
    AUDIO_ANALYZER_AVAILABLE = True
except ImportError:
    AUDIO_ANALYZER_AVAILABLE = False


API_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.1-8b-instant"
TIMEOUT = 60


def verificar_ffmpeg() -> bool:
    """Verifica si FFmpeg está instalado."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def verificar_dependencias() -> bool:
    """Verifica que las dependencias estén instaladas."""
    if not PYDUB_AVAILABLE:
        print("Error: pydub no está instalado")
        print("Ejecuta: pip install pydub")
        return False
    return True


def extraer_audio(video_path: str, audio_path: str = "audio.mp3") -> str:
    """Extrae el audio de un video usando FFmpeg."""
    print("[1/6] Extrayendo audio del video...")
    
    if os.path.exists(audio_path):
        os.remove(audio_path)
    
    comando = [
        "ffmpeg", "-y", "-i", video_path,
        "-q:a", "0", "-map", "a", audio_path
    ]
    
    try:
        subprocess.run(comando, capture_output=True, check=True)
        print(f"      Audio guardado: {audio_path}")
        return audio_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error al extraer audio: {e.stderr.decode()}")


def analizar_audio_video(audio_path: str = "audio.mp3", max_eventos: int = 30) -> List[Dict]:
    """Analiza el audio usando el módulo audio_analyzer."""
    print("[2/6] Analizando audio...")
    
    if not AUDIO_ANALYZER_AVAILABLE:
        print("      audio_analyzer no disponible")
        return []
    
    try:
        eventos = analizar_audio(audio_path, "audio.json")
        
        eventos = sorted(eventos, key=lambda x: x.get("intensidad", 0), reverse=True)
        eventos_top = eventos[:50]
        eventos_top = sorted(eventos_top, key=lambda x: x.get("timestamp", ""))
        
        print(f"      Eventos detectados: {len(eventos)}, usando top {len(eventos_top)}")
        return eventos_top
    except Exception as e:
        print(f"      Error en análisis: {e}")
        return []


def leer_transcripcion(ruta: str, max_chars: int = 8000) -> str:
    """Lee el archivo de transcripción."""
    print("[3/6] Leyendo transcripción...")
    with open(ruta, "r", encoding="utf-8") as f:
        contenido = f.read().strip()
    
    if len(contenido) > max_chars:
        contenido = contenido[:max_chars]
        print(f"      Transcripción truncada: {len(contenido)} caracteres")
    else:
        print(f"      Transcripción leída: {len(contenido)} caracteres")
    
    return contenido


def construir_prompt(transcripcion: str, datos_audio: List[Dict]) -> str:
    """Construye el prompt para la IA."""
    audio_json = json.dumps(datos_audio, indent=2, ensure_ascii=False)
    
    return f"""Eres un asistente que devuelve EXCLUSIVAMENTE código JSON válido. NO escribas texto antes ni después del JSON.

REGLAS ESTRICTAS:
- Tu respuesta DEBE empezar directamente con [
- Tu respuesta DEBE terminar directamente con ]
- NO escribas explicaciones, análisis ni texto adicional
- NO escribas markdown, ni código, ni nada fuera del JSON

Analiza esta transcripción y detecta los mejores clips virales.

DATOS DE AUDIO:
{audio_json}

TRANSCRIPCIÓN:
{transcripcion}

FORMATO JSON OBLIGATORIO (devuelve arrays con 3-10 elementos):
[
  {{
    "inicio": "00:00:00",
    "fin": "00:00:30",
    "duracion": 30,
    "score": 8,
    "intensidad": 7,
    "hook_score": 8,
    "audio_score": 7,
    "tipo": "gracioso",
    "hook": "inicio viral",
    "motivo": "razón del clip"
  }}
]

INSTRUCCIONES:
- Solo JSON válido
- Score mínimo 6
- Duración 15-60 segundos
- Máximo 10 clips
- Empieza con [ y termina con ]"""


def obtener_clips_ia(transcripcion: str, datos_audio: List[Dict]) -> List[Dict]:
    """Envía datos a la IA y retorna los clips detectados."""
    print("[4/6] Analizando con IA (Groq)...")
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Define GROQ_API_KEY")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    body = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": construir_prompt(transcripcion, datos_audio)
            }
        ]
    }
    
    try:
        respuesta = requests.post(
            API_ENDPOINT,
            headers=headers,
            json=body,
            timeout=TIMEOUT
        )
        if respuesta.status_code != 200:
            try:
                error_detail = respuesta.json().get("error", {}).get("message", respuesta.text)
            except:
                error_detail = respuesta.text
            raise ConnectionError(f"Error {respuesta.status_code}: {error_detail}")
    except requests.exceptions.Timeout:
        raise TimeoutError("La solicitud excedió el tiempo de espera")
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Error de conexión: {e}")
    
    try:
        datos = respuesta.json()
        contenido = datos["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Respuesta inválida de la API: {e}")
    
    print(f"      Respuesta IA ({len(contenido)} chars)")
    print(f"      === RESPUESTA COMPLETA ===")
    print(contenido[:1000])
    print(f"      === FIN RESPUESTA ===")
    
    json_match = re.search(r'\[[\s\S]*\]', contenido)
    if not json_match:
        print(f"      Preview: {contenido[:200]}...")
        raise ValueError("No se encontró JSON válido en la respuesta")
    
    clips = parsear_json_tolerante(contenido)
    if not clips:
        print(f"      JSON inválido, no se pudieron extraer clips")
        raise ValueError("No se pudo parsear la respuesta como JSON")
    
    print(f"      IA detectó {len(clips)} momentos")
    return clips


def timestamp_a_segundos(timestamp: str) -> int:
    """Convierte HH:MM:SS a segundos."""
    partes = list(map(int, timestamp.split(':')))
    while len(partes) < 3:
        partes.insert(0, 0)
    return partes[0] * 3600 + partes[1] * 60 + partes[2]


def reparar_json(texto: str) -> str:
    """Intenta reparar JSON común malformado."""
    texto = texto.strip()
    
    texto = re.sub(r"(\w+)\s*:\s*", r'"\1": ', texto)
    
    texto = re.sub(r":\s*'([^']*)'", r': "\1"', texto)
    
    texto = re.sub(r",(\s*[}\]])", r'\1', texto)
    
    texto = re.sub(r"(\d+)\s+(\d+:)", r'\1,\2', texto)
    
    return texto


def parsear_json_tolerante(texto: str) -> List[Dict]:
    """Intenta parsear JSON, intentando reparar errores comunes."""
    texto = texto.strip()
    
    first_bracket = texto.find('[')
    last_bracket = texto.rfind(']')
    
    if first_bracket == -1 or last_bracket == -1:
        json_str = texto
    else:
        json_str = texto[first_bracket:last_bracket+1]
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    try:
        reparado = reparar_json(json_str)
        return json.loads(reparado)
    except json.JSONDecodeError:
        pass
    
    try:
        objetos = []
        for match in re.finditer(r'\{[^{}]*\}', json_str):
            try:
                obj = json.loads(match.group())
                if "inicio" in obj and "fin" in obj:
                    objetos.append(obj)
            except:
                continue
        return objetos
    except:
        return []
    """Convierte segundos a HH:MM:SS."""
    horas = segundos // 3600
    minutos = (segundos % 3600) // 60
    segs = segundos % 60
    return f"{horas:02d}:{minutos:02d}:{segs:02d}"


def validar_clips(clips: List[Dict]) -> List[Dict]:
    """Valida y filtra los clips."""
    print("[5/6] Validando clips...")
    def convertir_ts(ts):
        """Convierte timestamp a segundos (maneja M:SS y HH:MM:SS)."""
        partes = ts.split(':')
        if len(partes) == 2:
            return int(partes[0]) * 60 + int(partes[1])
        elif len(partes) == 3:
            return int(partes[0]) * 3600 + int(partes[1]) * 60 + int(partes[2])
        return 0
    
    def formatear_ts(segundos):
        """Convierte segundos a HH:MM:SS."""
        h = segundos // 3600
        m = (segundos % 3600) // 60
        s = segundos % 60
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
    
    clips_validos = []
    
    for i, clip in enumerate(clips):
        if "inicio" not in clip or "fin" not in clip:
            continue
        
        score = clip.get("score", 7)
        if score < 6:
            continue
        
        inicio = str(clip["inicio"])
        fin = str(clip["fin"])
        
        try:
            inicio_seg = convertir_ts(inicio)
            fin_seg = convertir_ts(fin)
        except:
            continue
        
        duracion = fin_seg - inicio_seg
        
        if duracion < 15 or duracion > 60:
            continue
        
        clips_validos.append({
            "inicio": formatear_ts(inicio_seg),
            "fin": formatear_ts(fin_seg),
            "duracion": duracion,
            "score": score,
            "intensidad": clip.get("intensidad", 5),
            "hook_score": clip.get("hook_score", 5),
            "audio_score": clip.get("audio_score", 5),
            "tipo": clip.get("tipo", "otro"),
            "hook": clip.get("hook", ""),
            "motivo": clip.get("motivo", "")
        })
        print(f"      Clip {i+1}: {formatear_ts(inicio_seg)} - {formatear_ts(fin_seg)} ({duracion}s) | {clip.get('tipo', '?')} | Score: {score}")
    
    clips_validos.sort(key=lambda x: x["score"], reverse=True)
    clips_validos = clips_validos[:10]
    
    print(f"      Clips válidos: {len(clips_validos)}")
    return clips_validos


def crear_carpeta_salida(nombre: str = "clips") -> str:
    """Crea la carpeta de salida."""
    if not os.path.exists(nombre):
        os.makedirs(nombre)
    return nombre


def cortar_clip(video_entrada: str, inicio: str, fin: str, salida: str) -> bool:
    """Corta un clip usando FFmpeg (reencodificado para evitar freeze)."""
    try:
        comando = [
            "ffmpeg", "-y", "-ss", inicio, "-i", video_entrada,
            "-to", fin,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-avoid_negative_ts", "make_zero",
            salida
        ]
        subprocess.run(comando, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"      Error al cortar: {e.stderr.decode() if e.stderr else e}")
        return False


def procesar_clips(video_entrada: str, clips: List[Dict], carpeta: str) -> List[str]:
    """Genera todos los clips."""
    print("[6/6] Generando clips...")
    
    archivos = []
    info_clips = []
    
    for i, clip in enumerate(clips, 1):
        nombre = os.path.join(carpeta, f"clip_{i}.mp4")
        print(f"      Generando clip {i}: {clip['inicio']} - {clip['fin']}")
        
        if cortar_clip(video_entrada, clip["inicio"], clip["fin"], nombre):
            archivos.append(nombre)
            info_clips.append({
                "archivo": nombre,
                "inicio": clip["inicio"],
                "fin": clip["fin"]
            })
            print(f"      ✓ Guardado: {nombre}")
        else:
            print(f"      ✗ Falló")
    
    with open(os.path.join(carpeta, "clips_info.json"), "w", encoding="utf-8") as f:
        json.dump(info_clips, f, indent=2)
    
    return archivos


def main():
    """Función principal."""
    print("=" * 50)
    print("  CLIP GENERATOR - FULL AUTOMATIC")
    print("=" * 50)
    
    if len(sys.argv) != 3:
        print("\nUso: python main.py video.mp4 transcripcion.txt\n")
        sys.exit(1)
    
    video = sys.argv[1]
    transcripcion = sys.argv[2]
    
    if not os.path.exists(video):
        print(f"\nError: Video '{video}' no encontrado\n")
        sys.exit(1)
    
    if not os.path.exists(transcripcion):
        print(f"\nError: Transcripción '{transcripcion}' no encontrada\n")
        sys.exit(1)
    
    if not verificar_ffmpeg():
        print("\nError: FFmpeg no instalado\n")
        sys.exit(1)
    
    carpeta = crear_carpeta_salida("clips")
    print(f"\nCarpeta: {os.path.abspath(carpeta)}\n")
    
    try:
        audio_path = extraer_audio(video)
        
        datos_audio = []
        if AUDIO_ANALYZER_AVAILABLE:
            datos_audio = analizar_audio_video(audio_path)
        else:
            print("      Saltando análisis de audio (audio_analyzer no instalado)")
        
        texto = leer_transcripcion(transcripcion)
        clips_ia = obtener_clips_ia(texto, datos_audio)
        clips_validos = validar_clips(clips_ia)
        
        if not clips_validos:
            print("\nNo se encontraron clips válidos")
            sys.exit(0)
        
        archivos = procesar_clips(video, clips_validos, carpeta)
        
        print("\n" + "=" * 50)
        print(f"  CLIPS GENERADOS: {len(archivos)}")
        print(f"  UBICACIÓN: {os.path.abspath(carpeta)}")
        print("=" * 50)
        
    except ValueError as e:
        print(f"\nError de datos: {e}\n")
        sys.exit(1)
    except ConnectionError as e:
        print(f"\nError de conexión: {e}\n")
        sys.exit(1)
    except TimeoutError as e:
        print(f"\nError de tiempo: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

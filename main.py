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
MODEL = "llama-3.3-70b-versatile"
TIMEOUT = 120


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


def leer_transcripcion_completa(ruta: str) -> str:
    """Lee el archivo completo sin límite. Usar para formatear_transcripcion()."""
    with open(ruta, "r", encoding="utf-8") as f:
        return f.read().strip()


def leer_transcripcion_para_ia(ruta: str, max_chars: int = 6000) -> str:
    """Lee con límite de chars. Usar solo para construir el prompt de la IA."""
    with open(ruta, "r", encoding="utf-8") as f:
        contenido = f.read().strip()
    if len(contenido) > max_chars:
        contenido = contenido[:max_chars].rsplit('\n', 1)[0]
        print(f"      Transcripción truncada para IA: {len(contenido)} caracteres")
    return contenido


def leer_transcripcion(ruta: str, max_chars: int = 8000) -> str:
    """Lee el archivo de transcripción (deprecated, usar leer_transcripcion_completa o leer_transcripcion_para_ia)."""
    return leer_transcripcion_completa(ruta)


def formatear_transcripcion(transcripcion_raw: str, salida_path: str = "transcripcion_formatted.txt") -> str:
    """
    Convierte la transcripción cruda de YouTube al formato HH:MM:SS - Texto
    SIN modificar los timestamps ni el texto. Solo limpia y reformatea.
    Procesa TODAS las líneas sin límite.
    """
    import re
    
    print("[3.5/6] Formateando transcripción completa...")
    
    lineas_raw = transcripcion_raw.strip().split('\n')
    segmentos = []
    texto_pendiente = ""
    timestamp_actual = None
    
    RE_TS = re.compile(r'^(\d{1,2}:\d{2}(?::\d{2})?)')
    RE_SEG = re.compile(r'^\d+\s+(segundo|minuto|hora)s?', re.IGNORECASE)
    RE_MIN = re.compile(r'^\d+\s+minutos?\s+y\s+\d+\s+segundos?', re.IGNORECASE)
    
    for linea in lineas_raw:
        linea = linea.strip()
        if not linea:
            continue
        
        match_ts = RE_TS.match(linea)
        if match_ts:
            if texto_pendiente and timestamp_actual:
                h, m, s = _parse_timestamp(timestamp_actual)
                segmentos.append((f"{h:02d}:{m:02d}:{s:02d}", texto_pendiente))
            
            ts_str = match_ts.group(1)
            texto_restante = linea[len(match_ts.group(0)):].strip()
            
            texto_restante = RE_SEG.sub('', texto_restante)
            texto_restante = RE_MIN.sub('', texto_restante)
            
            timestamp_actual = ts_str
            texto_pendiente = texto_restante
        else:
            if texto_pendiente:
                texto_pendiente += " " + linea
    
    if texto_pendiente and timestamp_actual:
        h, m, s = _parse_timestamp(timestamp_actual)
        segmentos.append((f"{h:02d}:{m:02d}:{s:02d}", texto_pendiente))
    
    if not segmentos:
        print("      ✗ No se pudieron parsear segmentos")
        return transcripcion_raw
    
    with open(salida_path, "w", encoding="utf-8") as f:
        for ts, texto in segmentos:
            f.write(f"{ts} - {texto}\n")
    
    ts_primero = segmentos[0][0]
    ts_ultimo = segmentos[-1][0]
    print(f"      ✓ Transcripción formateada: {salida_path} ({len(segmentos)} segmentos)")
    print(f"      Rango: {ts_primero} hasta {ts_ultimo}")
    for i, (ts, txt) in enumerate(segmentos[:5]):
        print(f"        {ts}: {txt[:50]}...")
    
    return salida_path


def _parse_timestamp(ts: str) -> tuple:
    """Convierte timestamp M:SS o H:MM:SS a (h, m, s)."""
    partes = ts.split(':')
    if len(partes) == 2:
        return (0, int(partes[0]), int(partes[1]))
    elif len(partes) == 3:
        return (int(partes[0]), int(partes[1]), int(partes[2]))
    return (0, 0, 0)


def construir_prompt(transcripcion: str, datos_audio: List[Dict]) -> str:
    """Construye el prompt para la IA."""
    audio_json = json.dumps(datos_audio, indent=2, ensure_ascii=False)
    
    return f"""Sos un experto en contenido viral para TikTok, Instagram Reels y YouTube Shorts especializado en contenido deportivo y periodístico en español rioplatense. Tu tarea es analizar una transcripción y datos de audio para identificar los momentos con mayor potencial viral. Conocés en profundidad qué engancha a la audiencia hispanohablante de fútbol/deportes en redes sociales. DEVOLVÉ ÚNICAMENTE UN ARRAY JSON VÁLIDO. Sin texto antes ni después. Sin markdown. Empezá con [ y terminá con ].

CRITERIOS DE VIRALIDAD (ordenados por peso):

REVELACIÓN O EXCLUSIVA (peso 10/10)
Información que el espectador no sabía y genera sorpresa. Datos internos, rumores confirmados, primicias. Frases como "tengo la información", "me dijeron", "la verdad es que". Ejemplo: revelar por qué realmente salió un jugador, una lesión real vs oficial.

CONTROVERSIA O CONTRADICCIÓN (peso 9/10)
Momentos donde se contradice la versión oficial. Críticas directas a jugadores, cuerpo técnico o dirigencia. Predicciones arriesgadas con fundamento. Frases como "no puede ser", "me parece mal", "error garrafal".

DATO CONCRETO IMPACTANTE (peso 8/10)
Números, fechas, montos específicos que sorprenden. Comparaciones que reencuadran la realidad. Ejemplo: "costó más que Julián Álvarez", "8 millones de euros".

MOMENTO EMOCIONAL AUTÉNTICO (peso 7/10)
Nostalgia, pasión, anécdota personal del conductor. Momento donde el conductor se quiebra o se emociona. Frases como "se me pone la piel de gallina", "lo atesoro".

ANÁLISIS TÉCNICO SIMPLE Y CLARO (peso 6/10)
Explicación de por qué pasó algo en cancha de manera accesible. Máximo cuando usa metáforas o ejemplos concretos. Ejemplo: explicar por qué sacaron a un jugador con causa táctica clara.

PREDICCIÓN O ANTICIPACIÓN (peso 5/10)
El conductor adelanta algo que va a pasar. Spoiler de una decisión técnica o dirigencial futura. Genera curiosidad sobre si acertó o no.

SEÑALES DE AUDIO QUE AMPLIFICAN EL SCORE:
- intensidad > 7.0 en el timestamp: +1 punto al score
- evento "cambio_brusco" cerca del inicio del clip: +0.5 puntos
- evento "momento_intenso" sostenido: +1 punto
- intensidad < 4.0 (pausa dramática antes del clip): +0.5 puntos

REGLAS PARA DEFINIR INICIO Y FIN DE CADA CLIP:

INICIO del clip:
- Empezar 2-3 segundos ANTES de la frase clave para dar contexto
- Nunca cortar en medio de una oración
- Preferir iniciar después de una pausa natural (silencio en datos de audio)
- El primer segundo debe enganchar: pregunta, dato impactante o afirmación fuerte

FIN del clip:
- Terminar en punto final de idea, nunca a mitad de frase
- Ideal: terminar con conclusión o frase de impacto
- Dejar al espectador con ganas de saber más (cliffhanger) cuando sea posible
- Máximo 60 segundos, mínimo 20 segundos
- El punto óptimo es 30-45 segundos para TikTok/Reels

EVITAR:
- Clips que empiezan con "y", "pero", "entonces" sin contexto previo
- Clips que terminan con "..." o frase inconclusa sin intención
- Dos clips con el mismo tema o personaje
- Clips de más de 55 segundos salvo que sea una revelación excepcional

DATOS DE AUDIO DEL VIDEO:
{audio_json}

Instrucciones para usar los datos de audio:
- Buscá timestamps con intensidad >= 7.5: son momentos de alta energía vocal
- Los eventos "cambio_brusco" marcan transiciones abruptas de tema
- Los eventos "momento_intenso" indican energía sostenida (bueno para clips)
- Correlacioná los timestamps del audio con los timestamps de la transcripción para validar que los clips propuestos coincidan con momentos de energía alta

TRANSCRIPCIÓN COMPLETA:
{transcripcion}

FORMATO JSON DE SALIDA OBLIGATORIO:
Devolvé entre 5 y 10 clips ordenados de MAYOR a MENOR score.
[
{{
"inicio": "HH:MM:SS",
"fin": "HH:MM:SS",
"duracion": 35,
"score": 9,
"criterio_principal": "revelacion",
"titulo_sugerido": "La VERDAD de por qué sacaron a Bustos",
"hook_texto": "Lo que nadie te contó",
"primer_segundo": "frase exacta con la que arranca el clip",
"motivo": "explicación de por qué es viral en 1 oración",
"tipo_contenido": "exclusiva",
"audio_score": 8
}}

CAMPOS OBLIGATORIOS:
- inicio/fin: timestamps exactos en HH:MM:SS
- duracion: segundos enteros (fin - inicio)
- score: 1-10 basado en criterios de viralidad
- criterio_principal: uno de [revelacion, controversia, dato_impactante, emocional, analisis_tactico, prediccion]
- titulo_sugerido: título para el clip en español rioplatense, max 60 chars, sin clickbait vacío, con el dato concreto que engancha
- hook_texto: texto del gancho inicial (2-5 palabras), distinto para cada clip
- primer_segundo: primeras palabras exactas de la transcripción que abre el clip
- motivo: por qué este momento específico va a generar engagement
- tipo_contenido: uno de [exclusiva, analisis, emocion, prediccion, controversia]
- audio_score: 1-10 basado en los datos de audio en ese rango de tiempo

REGLAS FINALES:
- Score mínimo para incluir: 7
- Máximo 10 clips
- No repetir el mismo tema o personaje en más de 2 clips
- El primer clip del array DEBE ser el de mayor score
- Todos los timestamps deben existir en la transcripción proporcionada
- Empezá con [ y terminá con ]"""


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


def timestamp_a_segundos(timestamp: str) -> float:
    """Convierte timestamp a segundos (maneja HH:MM:SS, MM:SS, o segundos)."""
    timestamp = str(timestamp).strip()
    
    try:
        if ':' in timestamp:
            partes = timestamp.split(':')
            if len(partes) == 2:
                return int(partes[0]) * 60 + float(partes[1])
            elif len(partes) == 3:
                return int(partes[0]) * 3600 + int(partes[1]) * 60 + float(partes[2])
        else:
            return float(timestamp)
    except ValueError:
        return 0.0
    
    return 0.0


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
        partes = ts.split(':')
        if len(partes) == 2:
            return int(partes[0]) * 60 + int(partes[1])
        elif len(partes) == 3:
            return int(partes[0]) * 3600 + int(partes[1]) * 60 + int(partes[2])
        return 0
    
    def formatear_ts(segundos):
        h = segundos // 3600
        m = (segundos % 3600) // 60
        s = segundos % 60
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
    
    clips_validos = []
    
    for i, clip in enumerate(clips):
        if "inicio" not in clip or "fin" not in clip:
            continue
        
        score = clip.get("score", 7)
        if score < 7:
            continue
        
        inicio = str(clip["inicio"])
        fin = str(clip["fin"])
        
        try:
            inicio_seg = convertir_ts(inicio)
            fin_seg = convertir_ts(fin)
        except:
            continue
        
        duracion = fin_seg - inicio_seg
        
        if duracion < 20 or duracion > 60:
            continue
        
        criterio = clip.get("criterio_principal", "otro")
        titulo = clip.get("titulo_sugerido", "")
        
        clips_validos.append({
            "inicio": formatear_ts(inicio_seg),
            "fin": formatear_ts(fin_seg),
            "duracion": duracion,
            "score": score,
            "intensidad": clip.get("intensidad", 5),
            "audio_score": clip.get("audio_score", 5),
            "criterio_principal": criterio,
            "titulo_sugerido": titulo,
            "hook_texto": clip.get("hook_texto", ""),
            "primer_segundo": clip.get("primer_segundo", ""),
            "motivo": clip.get("motivo", ""),
            "tipo_contenido": clip.get("tipo_contenido", "otro"),
        })
        print(f"      Clip {i+1}: {formatear_ts(inicio_seg)} - {formatear_ts(fin_seg)} ({duracion}s) | {criterio} | Score: {score} | {titulo[:40]}")
    
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
    """Corta un clip usando FFmpeg."""
    inicio_seg = timestamp_a_segundos(inicio)
    fin_seg = timestamp_a_segundos(fin)
    duracion = int(fin_seg - inicio_seg)
    
    if duracion <= 0:
        print(f"      Duración inválida: {duracion}s (inicio={inicio}, fin={fin})")
        return False
    
    print(f"      Corte: {inicio} -> {fin} (duración: {duracion}s)")
    
    try:
        comando = [
            "ffmpeg", "-y",
            "-ss", str(inicio),
            "-i", video_entrada,
            "-t", str(duracion),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart",
            "-progress", "pipe:1",
            salida
        ]
        result = subprocess.run(comando, capture_output=True, timeout=120)
        if result.returncode != 0:
            print(f"      FFmpeg error: {result.stderr.decode()[:200]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"      Timeout al cortar clip")
        return False
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
        inicio = str(clip["inicio"])
        fin = str(clip["fin"])
        print(f"      Clip {i}: inicio={inicio}, fin={fin}")
        
        if cortar_clip(video_entrada, inicio, fin, nombre):
            if os.path.exists(nombre):
                tamano = os.path.getsize(nombre)
                print(f"      ✓ Guardado: {nombre} ({tamano} bytes)")
            else:
                print(f"      ✗ Archivo no creado")
            archivos.append(nombre)
            info_clips.append({
                "archivo": nombre,
                "inicio": clip["inicio"],
                "fin": clip["fin"],
                "titulo_sugerido": clip.get("titulo_sugerido", ""),
                "hook_texto": clip.get("hook_texto", ""),
                "criterio_principal": clip.get("criterio_principal", ""),
                "score": clip.get("score", 0),
                "primer_segundo": clip.get("primer_segundo", ""),
                "motivo": clip.get("motivo", ""),
            })
        else:
            print(f"      ✗ Falló")
    
    with open(os.path.join(carpeta, "clips_info.json"), "w", encoding="utf-8") as f:
        json.dump(info_clips, f, indent=2, ensure_ascii=False)
    
    return archivos


def main():
    """Función principal."""
    print("=" * 50)
    print("  CLIP GENERATOR")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("\nUso:")
        print("  Con transcripción manual: python main.py video.mp4 transcripcion.txt")
        print("  Modo automático (Whisper): python main.py video.mp4")
        print("  Solo mejor clip: python main.py video.mp4 --single")
        print("  Corte manual: python main.py video.mp4 00:01:00 00:01:30")
        print("\nEjemplos:")
        print("  python main.py video.mp4 transcripcion.txt")
        print("  python main.py video.mp4  (usa Whisper automático)")
        print("  python main.py video.mp4 --single")
        print("=" * 50)
        sys.exit(1)
    
    video = sys.argv[1]
    
    if not os.path.exists(video):
        print(f"\nError: Video '{video}' no encontrado\n")
        sys.exit(1)
    
    if not verificar_ffmpeg():
        print("\nError: FFmpeg no instalado\n")
        sys.exit(1)
    
    carpeta = crear_carpeta_salida("clips")
    print(f"\nCarpeta: {os.path.abspath(carpeta)}\n")
    
    modo_whisper = False
    transcripcion_path = None
    modo_single = False
    
    if len(sys.argv) >= 4:
        if sys.argv[3] == "--single":
            transcripcion_path = sys.argv[2]
            modo_single = True
        else:
            inicio = sys.argv[2]
            fin = sys.argv[3]
            print(f"MODO MANUAL: {inicio} -> {fin}")
            print("=" * 50)
            
            nombre = os.path.join(carpeta, "clip_single.mp4")
            if cortar_clip(video, inicio, fin, nombre):
                if os.path.exists(nombre):
                    tamano = os.path.getsize(nombre)
                    info_clips = [{"archivo": nombre, "inicio": inicio, "fin": fin}]
                    with open(os.path.join(carpeta, "clips_info.json"), "w", encoding="utf-8") as f:
                        json.dump(info_clips, f, indent=2)
                    print(f"\n✓ Clip guardado: {nombre} ({tamano} bytes)")
                    print(f"  Editar con: python editor_viral.py {nombre} clips_editados")
                else:
                    print("\n✗ Error: archivo no creado")
            else:
                print("\n✗ Error al cortar clip")
            sys.exit(0)
    elif len(sys.argv) >= 3:
        transcripcion_path = sys.argv[2]
        if sys.argv[2] == "--single":
            modo_single = True
            transcripcion_path = None
        elif not os.path.exists(transcripcion_path):
            print(f"\nError: Transcripción '{transcripcion_path}' no encontrada\n")
            sys.exit(1)
    else:
        modo_whisper = True
    
    print("MODO AUTOMÁTICO")
    if modo_single:
        print("  (Solo el mejor clip)")
    if modo_whisper:
        print("  (Transcripción con Whisper)")
    print("=" * 50)
    
    try:
        audio_path = extraer_audio(video)
        
        datos_audio = []
        if AUDIO_ANALYZER_AVAILABLE:
            datos_audio = analizar_audio_video(audio_path)
        else:
            print("      Saltando análisis de audio (audio_analyzer no instalado)")
        
        if transcripcion_path and os.path.exists(transcripcion_path):
            from whisper_transcriber import parsear_transcripcion_youtube
            print("\n[3/6] Parseando transcripción...")
            parsear_transcripcion_youtube(
                transcripcion_path,
                "transcripcion_formatted.txt"
            )
            texto_para_ia = leer_transcripcion_para_ia("transcripcion_formatted.txt")
            print("      Modo: transcripción manual (instantáneo)")
        else:
            from whisper_transcriber import transcribir_video
            print("\n[3/6] Transcribiendo con faster-whisper...")
            print("      Modelo: large-v3-turbo (4x más rápido)")
            transcribir_video(
                video,
                "transcripcion_formatted.txt"
            )
            texto_para_ia = leer_transcripcion_para_ia("transcripcion_formatted.txt")
            print("      Modo: faster-whisper automático")
        
        if texto_para_ia:
            clips_ia = obtener_clips_ia(texto_para_ia, datos_audio)
        else:
            clips_ia = []
        
        clips_validos = validar_clips(clips_ia)
        
        if not clips_validos:
            print("\nNo se encontraron clips válidos")
            sys.exit(0)
        
        if modo_single:
            print(f"\nProcesando solo el mejor clip (score: {clips_validos[0]['score']})")
            mejor_clip = [clips_validos[0]]
            archivos = procesar_clips(video, mejor_clip, carpeta)
        else:
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

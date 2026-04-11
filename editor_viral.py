#!/usr/bin/env python3
"""
Editor profesional de clips para contenido viral.
Convierte clips básicos en videos listos para TikTok, Shorts y Reels.
"""

import os
import json
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)


@dataclass
class ClipEditado:
    """Representa un clip procesado."""
    ruta_original: str
    ruta_salida: str
    subtitulos: List[Dict]
    momentos_clave: List[Dict]


class VideoEditor:
    """Editor profesional de videos virales."""
    
    def __init__(self, clips_dir: str = "clips", salida_dir: str = "clips_editados"):
        self.clips_dir = Path(clips_dir)
        self.salida_dir = Path(salida_dir)
        self.salida_dir.mkdir(exist_ok=True)
        
        self.ancho_final = 1080
        self.alto_final = 1920
        self.fps = 30
        
        self._verificar_dependencias()
    
    def _verificar_dependencias(self):
        """Verifica que las herramientas estén disponibles."""
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True)
            log.info("  FFmpeg: OK")
        except:
            log.error("  FFmpeg no encontrado")
        
        try:
            import moviepy
            log.info("  MoviePy: OK")
        except ImportError:
            log.warning("  MoviePy no instalado")
        
        try:
            import cv2
            log.info("  OpenCV: OK")
        except ImportError:
            log.warning("  OpenCV no instalado")
    
    def cargar_clips(self) -> List[Path]:
        """Carga todos los clips de la carpeta."""
        log.info("[1/8] Cargando clips...")
        clips = list(self.clips_dir.glob("*.mp4"))
        log.info(f"      Clips encontrados: {len(clips)}")
        return clips
    
    def obtener_info_video(self, ruta: str) -> Dict:
        """Obtiene información del video usando ffprobe."""
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", ruta
        ]
        resultado = subprocess.run(cmd, capture_output=True, text=True)
        return json.loads(resultado.stdout)
    
    def detectar_rostro(self, frame) -> Tuple[int, int, int, int]:
        """Detecta rostro en un frame usando OpenCV."""
        try:
            import cv2
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                return x, y, w, h
            
        except Exception as e:
            log.warning(f"      Error detectando rostro: {e}")
        
        return None
    
    def calcular_crop_vertical(
        self, 
        ancho_orig: int, 
        alto_orig: int, 
        rostro: Tuple = None
    ) -> Tuple[int, int, int, int]:
        """Calcula coordenadas para crop a 9:16 centrado en rostro."""
        ratio_objetivo = self.ancho_final / self.alto_final
        
        if ancho_orig / alto_orig > ratio_objetivo:
            nuevo_ancho = int(alto_orig * ratio_objetivo)
            x_offset = 0
            y_offset = 0
        else:
            nuevo_alto = int(ancho_orig / ratio_objetivo)
            nuevo_ancho = ancho_orig
            x_offset = 0
            y_offset = 0
        
        if rostro:
            cx, cy = rostro[0] + rostro[2]//2, rostro[1] + rostro[3]//2
            x_offset = max(0, min(cx - nuevo_ancho//2, ancho_orig - nuevo_ancho))
            y_offset = max(0, min(cy - self.alto_final//2, alto_orig - self.alto_final))
        
        x_offset = max(0, x_offset)
        y_offset = max(0, y_offset)
        
        return x_offset, y_offset, nuevo_ancho, self.alto_final
    
    def generar_subtitulos_basico(self, video_path: str) -> List[Dict]:
        """Genera subtítulos básicos sincronizados (sin Whisper)."""
        log.info("[2/8] Generando subtítulos...")
        
        try:
            import cv2
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duracion = total_frames / fps
            
            subtitulos = []
            num_subtitulos = int(duracion / 3)
            
            for i in range(num_subtitulos):
                inicio = i * 3
                fin = min((i + 1) * 3, duracion)
                
                subtitulos.append({
                    "inicio": inicio,
                    "fin": fin,
                    "texto": f"Momento {i + 1}",
                    "palabras": [{"palabra": f"Palabra {j+1}", "start": inicio + j*0.5} for j in range(min(4, int(fin-inicio)*2))]
                })
            
            cap.release()
            log.info(f"      Subtítulos generados: {len(subtitulos)}")
            return subtitulos
            
        except Exception as e:
            log.error(f"      Error generando subtítulos: {e}")
            return []
    
    def detectar_momentos_clave(self, video_path: str) -> List[Dict]:
        """Detecta momentos clave basado en análisis de audio."""
        log.info("[3/8] Detectando momentos clave...")
        
        momentos = []
        
        try:
            audio_path = "temp_audio.wav"
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                audio_path
            ], capture_output=True)
            
            try:
                import librosa
                import numpy as np
                
                y, sr = librosa.load(audio_path, sr=16000)
                rms = librosa.feature.rms(y=y)[0]
                
                media = np.mean(rms)
                std = np.std(rms)
                umbral = media + 1.5 * std
                
                duracion = len(y) / sr
                frame_time = duracion / len(rms)
                
                for i, val in enumerate(rms):
                    if val > umbral:
                        t = i * frame_time
                        intensidad = min(10, (val / umbral) * 5 + 5)
                        
                        momentos.append({
                            "timestamp": t,
                            "tipo": "intenso",
                            "intensidad": intensidad
                        })
                
                os.remove(audio_path)
                
            except ImportError:
                log.warning("      librosa no disponible, usando detección básica")
                momentos = [{"timestamp": 0, "tipo": "inicio", "intensidad": 8}]
                
        except Exception as e:
            log.warning(f"      Error en análisis de audio: {e}")
            momentos = [{"timestamp": 0, "tipo": "inicio", "intensidad": 8}]
        
        log.info(f"      Momentos detectados: {len(momentos)}")
        return momentos[:20]
    
    def crear_hook_inicial(self, momento_clave: Dict) -> str:
        """Crea texto de hook basado en el momento."""
        hooks = [
            "MIRÁ ESTO 🤯",
            "NO PUEDE SER",
            "LO QUE PASÓ DESPUÉS",
            "INCREÍBLE",
            "QUÉ PASSÓ?"
        ]
        return hooks[0]
    
    def aplicar_formato_vertical(self, video_path: str, salida: str) -> bool:
        """Convierte video a formato vertical 9:16."""
        log.info("[4/8] Aplicando formato vertical...")
        
        try:
            info = self.obtener_info_video(video_path)
            streams = info.get('streams', [])
            
            ancho = 1920
            alto = 1080
            
            for stream in streams:
                if stream.get('codec_type') == 'video':
                    ancho = int(stream.get('width', 1920))
                    alto = int(stream.get('height', 1080))
                    break
            
            ratio = ancho / alto
            ratio_objetivo = 9 / 16
            
            if abs(ratio - ratio_objetivo) < 0.1:
                cmd = [
                    "ffmpeg", "-y", "-i", video_path,
                    "-vf", f"scale={self.ancho_final}:{self.alto_final}",
                    "-c:a", "copy",
                    "-r", str(self.fps),
                    salida
                ]
            else:
                if ratio > ratio_objetivo:
                    nuevo_ancho = int(alto * ratio_objetivo)
                    crop_x = (ancho - nuevo_ancho) // 2
                    filtro = f"crop={nuevo_ancho}:{alto}:{crop_x}:0,scale={self.ancho_final}:{self.alto_final}"
                else:
                    nuevo_alto = int(ancho / ratio_objetivo)
                    crop_y = (alto - nuevo_alto) // 2
                    filtro = f"crop={ancho}:{nuevo_alto}:0:{crop_y},scale={self.ancho_final}:{self.alto_final}"
                
                cmd = [
                    "ffmpeg", "-y", "-i", video_path,
                    "-vf", filtro,
                    "-c:a", "copy",
                    "-r", str(self.fps),
                    salida
                ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            log.info(f"      Video vertical creado")
            return True
            
        except subprocess.CalledProcessError as e:
            log.error(f"      Error: {e.stderr.decode() if e.stderr else e}")
            return False
    
    def agregar_subtitulos_virales_ffmpeg(self, video_path: str, subtitulos: List[Dict], salida: str) -> bool:
        """Agrega subtítulos estilo viral usando método mejorado."""
        log.info("[5/8] Agregando subtítulos virales...")
        
        if not subtitulos:
            return False
        
        try:
            filtros = []
            
            for sub in subtitulos[:15]:
                inicio = sub.get('inicio', 0)
                fin = sub.get('fin', inicio + 3)
                texto = sub.get('texto', '').replace("'", "\\'").replace(":", "\\:")
                
                filtro = (
                    f"drawtext=text='{texto}':"
                    f"fontsize=64:fontcolor=white@0.95:"
                    f"borderw=5:bordercolor=black@0.7:"
                    f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
                    f"x=(w-text_w)/2:y=h-280:"
                    f"enable='between(t,{inicio},{fin})'"
                )
                filtros.append(filtro)
            
            if filtros:
                filtro_final = ",".join(filtros)
                cmd = [
                    "ffmpeg", "-y", "-i", video_path,
                    "-vf", filtro_final,
                    "-c:a", "copy",
                    "-preset", "fast",
                    salida
                ]
                subprocess.run(cmd, capture_output=True, check=True)
                log.info(f"      Subtítulos virales aplicados")
                return True
            
        except Exception as e:
            log.warning(f"      Error en subtítulos: {e}")
        
        return False
    
    def agregar_hook_video(self, video_path: str, hook_text: str, salida: str) -> bool:
        """Agrega hook inicial al video."""
        log.info("[6/8] Agregando hook inicial...")
        
        try:
            filtro = (
                f"drawtext=text='{hook_text}':"
                f"fontsize=64:fontcolor=yellow:borderw=4:bordercolor=black:"
                f"x=(w-text_w)/2:y=h/2-100:"
                f"enable='between(t,0,3)'"
            )
            
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", filtro,
                "-c:a", "copy",
                "-t", "60",
                salida
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            log.info(f"      Hook agregado")
            return True
            
        except Exception as e:
            log.warning(f"      Error agregando hook: {e}")
            return False
    
    def mejorar_audio(self, video_path: str, salida: str) -> bool:
        """Mejora el audio del video."""
        log.info("[7/8] Mejorando audio...")
        
        try:
            filtro = "loudnorm=I=-16:TP=-1.5:LRA=11"
            
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-af", filtro,
                "-c:v", "copy",
                salida
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            log.info(f"      Audio mejorado")
            return True
            
        except Exception as e:
            log.warning(f"      Error mejorando audio: {e}")
            return False
    
    def exportar_final(self, video_path: str, salida: str) -> bool:
        """Exporta video final optimizado."""
        log.info("[8/8] Exportando video final...")
        
        try:
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-c:v", "libx264", "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart",
                salida
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            log.info(f"      Exportado: {os.path.basename(salida)}")
            return True
            
        except Exception as e:
            log.error(f"      Error en exportación: {e}")
            return False
    
    def procesar_clip(self, clip_path: Path, index: int, transcripcion_path: str = None) -> Optional[str]:
        """Procesa un clip individual."""
        log.info(f"\n{'='*40}")
        log.info(f"  PROCESANDO CLIP {index + 1}")
        log.info(f"{'='*40}")
        
        nombre = clip_path.stem
        temp_vertical = self.salida_dir / f"{nombre}_temp.mp4"
        temp_subs = self.salida_dir / f"{nombre}_subs.mp4"
        salida_final = self.salida_dir / f"{nombre}_viral.mp4"
        
        try:
            if not self.aplicar_formato_vertical(str(clip_path), str(temp_vertical)):
                return None
            
            momentos = self.detectar_momentos_clave(str(clip_path))
            
            if transcripcion_path and os.path.exists(transcripcion_path):
                from sub_reales import crear_video_con_subtitulos, leer_transcripcion, ts_a_segundos
                import json as json_lib
                
                log.info(f"[5/8] Agregando subtítulos desde: {transcripcion_path}")
                
                # Leer info de clips
                info_path = self.clips_dir / "clips_info.json"
                inicio_clip = 0
                duracion_clip = 60
                
                if info_path.exists():
                    with open(info_path, 'r') as f:
                        info_clips = json_lib.load(f)
                    if index < len(info_clips):
                        info = info_clips[index]
                        inicio_clip = ts_a_segundos(info['inicio'])
                        fin_clip = ts_a_segundos(info['fin'])
                        duracion_clip = fin_clip - inicio_clip
                        log.info(f"      Clip: {info['inicio']} - {info['fin']} ({duracion_clip}s)")
                
                segmentos = leer_transcripcion(transcripcion_path)
                log.info(f"      Segmentos leídos: {len(segmentos)}")
                
                if segmentos:
                    crear_video_con_subtitulos(
                        str(temp_vertical), 
                        segmentos, 
                        str(temp_subs), 
                        inicio_clip,
                        duracion_clip
                    )
                    temp_subs = temp_vertical
                else:
                    log.warning("      Sin segmentos - saltando")
                    temp_subs = temp_vertical
            else:
                from subtitulos_virales import crear_clip_con_efectos
                subtitulos = self.generar_subtitulos_basico(str(clip_path))
                hook = self.crear_hook_inicial(momentos[0] if momentos else None)
                
                crear_clip_con_efectos(
                    str(temp_vertical),
                    str(temp_subs),
                    subtitulos,
                    momentos,
                    hook
                )
            
            self.mejorar_audio(str(temp_subs), str(salida_final))
            
            for temp in [temp_vertical, temp_subs]:
                if temp.exists() and temp != salida_final:
                    try:
                        temp.unlink()
                    except:
                        pass
            
            log.info(f"      ✓ Clip viral: {os.path.basename(salida_final)}")
            return str(salida_final)
            
        except Exception as e:
            log.error(f"      Error procesando clip: {e}")
            return None
    
    def procesar_todos(self, transcripcion_path: str = None) -> List[str]:
        """Procesa todos los clips."""
        log.info("\n" + "=" * 50)
        log.info("  EDITOR DE VIDEOS VIRALES")
        log.info("=" * 50)
        
        clips = self.cargar_clips()
        
        if not clips:
            log.warning("No se encontraron clips")
            return []
        
        resultados = []
        
        for i, clip in enumerate(clips):
            resultado = self.procesar_clip(clip, i, transcripcion_path)
            if resultado:
                resultados.append(resultado)
        
        log.info("\n" + "=" * 50)
        log.info(f"  PROCESO COMPLETADO")
        log.info(f"  Videos editados: {len(resultados)}")
        log.info(f"  Salida: {self.salida_dir}")
        log.info("=" * 50)
        
        return resultados


def main():
    """Función principal."""
    import sys
    
    clips_dir = sys.argv[1] if len(sys.argv) > 1 else "clips"
    salida_dir = sys.argv[2] if len(sys.argv) > 2 else "clips_editados"
    transcripcion = sys.argv[3] if len(sys.argv) > 3 else None
    
    editor = VideoEditor(clips_dir, salida_dir)
    resultados = editor.procesar_todos(transcripcion)
    
    for r in resultados:
        print(f"  - {r}")


if __name__ == "__main__":
    main()

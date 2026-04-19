#!/usr/bin/env python3
"""
Editor de clips para contenido viral.
Convierte clips básicos en videos listos para TikTok, Shorts y Reels.
Solo formato vertical + mejora de audio - sin subtítulos.
"""

import os
import json
import subprocess
import logging
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)


class VideoEditor:
    """Editor de videos virales."""
    
    def __init__(self, clips_dir: str = "clips", salida_dir: str = "clips_editados"):
        self.clips_dir = Path(clips_dir)
        self.salida_dir = Path(salida_dir)
        self.salida_dir.mkdir(exist_ok=True)
        
        self.ancho_final = 1080
        self.alto_final = 1920
        self.fps = 30
        
        self._verificar_dependencias()
    
    def _verificar_dependencias(self):
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True)
            log.info("  FFmpeg: OK")
        except:
            log.error("  FFmpeg no encontrado")
        
        try:
            import cv2
            log.info("  OpenCV: OK")
        except ImportError:
            log.warning("  OpenCV no instalado")
    
    def cargar_clips(self) -> List[Path]:
        log.info("[1/4] Cargando clips...")
        clips = list(self.clips_dir.glob("*.mp4"))
        log.info(f"      Clips encontrados: {len(clips)}")
        return clips
    
    def detectar_rostro(self, frame):
        """Detecta rostro con OpenCV."""
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
                return faces[0]
        except Exception:
            pass
        return None
    
    def aplicar_formato_vertical(self, video_path: str, salida: str) -> bool:
        log.info("[2/4] Aplicando formato vertical...")
        
        try:
            import cv2
            import numpy as np
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            if ancho <= 0 or alto <= 0:
                log.error("      No se pudo leer dimensiones")
                return False
            
            posiciones_x = []
            posiciones_y = []
            
            cap = cv2.VideoCapture(video_path)
            sample_interval = max(1, int(fps * 3))
            frame_idx = 0
            frames_procesados = 0
            
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            if not face_cascade.empty():
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % sample_interval == 0:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(
                            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
                        )
                        if len(faces) > 0:
                            x, y, w, h = faces[0]
                            cx = x + w // 2
                            cy = y + h // 2
                            posiciones_x.append(cx)
                            posiciones_y.append(cy)
                            frames_procesados += 1
                    frame_idx += 1
                cap.release()
            
            ratio_objetivo = 9 / 16
            ancho_crop = int(alto * ratio_objetivo)
            
            if posiciones_x and frames_procesados > 0:
                cx_promedio = np.mean(posiciones_x)
                crop_x = int(cx_promedio - ancho_crop / 2)
                crop_x = max(0, min(crop_x, ancho - ancho_crop))
                log.info(f"      Cara detectada en {frames_procesados} frames")
            else:
                crop_x = (ancho - ancho_crop) // 2
                log.warning("      Sin cara - usando crop centrado")
            
            filtro = (
                f"crop={ancho_crop}:{alto}:{crop_x}:0,"
                f"scale={self.ancho_final}:{self.alto_final}"
            )
            
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path,
                "-vf", filtro,
                "-c:v", "libx264", "-preset", "fast", "-crf", "17",
                "-c:a", "aac", "-b:a", "192k",
                "-r", str(self.fps),
                "-movflags", "+faststart",
                salida
            ], capture_output=True, check=True)
            
            log.info("      Video vertical creado")
            return True
            
        except Exception as e:
            log.error(f"      Error: {e}")
            return False
    
    def mejorar_audio(self, video_path: str, salida: str) -> bool:
        log.info("[3/4] Mejorando audio...")
        
        try:
            filtro = "loudnorm=I=-16:TP=-1.5:LRA=11"
            
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path,
                "-af", filtro,
                "-c:v", "copy",
                "-preset", "ultrafast",
                salida
            ], capture_output=True, check=True)
            
            log.info("      Audio mejorado")
            return True
            
        except Exception as e:
            log.warning(f"      Error mejorando audio: {e}")
            return False
    
    def procesar_clip(self, clip_path: Path, index: int) -> Optional[str]:
        """Procesa un clip individual: vertical + audio."""
        log.info(f"\n{'='*40}")
        log.info(f"  PROCESANDO CLIP {index + 1}")
        log.info(f"{'='*40}")
        
        nombre = clip_path.stem
        temp_vertical = self.salida_dir / f"{nombre}_temp.mp4"
        salida_final = self.salida_dir / f"{nombre}_editado.mp4"
        
        info_path = self.clips_dir / "clips_info.json"
        inicio_clip = 0.0
        duracion_clip = 60.0
        titulo = ""
        criterio = ""
        
        if info_path.exists():
            with open(info_path, 'r') as f:
                info_clips = json.load(f)
            if index < len(info_clips):
                info = info_clips[index]
                inicio_clip = self._ts_a_segundos(info['inicio'])
                fin_clip = self._ts_a_segundos(info['fin'])
                duracion_clip = fin_clip - inicio_clip
                titulo = info.get('titulo_sugerido', '')
                criterio = info.get('criterio_principal', '')
                
                if titulo:
                    log.info(f"      Título: {titulo}")
                if criterio:
                    log.info(f"      Tipo: {criterio}")
        
        try:
            if not self.aplicar_formato_vertical(str(clip_path), str(temp_vertical)):
                return None
            
            self.mejorar_audio(str(temp_vertical), str(salida_final))
            
            if temp_vertical.exists():
                try:
                    temp_vertical.unlink()
                except:
                    pass
            
            log.info(f"      ✓ Clip editado: {os.path.basename(salida_final)}")
            return str(salida_final)
            
        except Exception as e:
            log.error(f"      Error: {e}")
            return None
    
    def _ts_a_segundos(self, ts: str) -> float:
        partes = ts.split(':')
        while len(partes) < 3:
            partes.insert(0, '0')
        return int(partes[0]) * 3600 + int(partes[1]) * 60 + float(partes[2])
    
    def procesar_todos(self) -> List[str]:
        log.info("\n" + "=" * 50)
        log.info("  EDITOR DE VIDEOS VIRALES")
        log.info("=" * 50)
        
        clips = self.cargar_clips()
        
        if not clips:
            log.warning("No se encontraron clips")
            return []
        
        resultados = []
        for i, clip in enumerate(clips):
            resultado = self.procesar_clip(clip, i)
            if resultado:
                resultados.append(resultado)
        
        log.info("\n" + "=" * 50)
        log.info(f"  PROCESO COMPLETADO")
        log.info(f"  Videos editados: {len(resultados)}")
        log.info(f"  Salida: {self.salida_dir}")
        log.info("=" * 50)
        
        return resultados


def main():
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("\nEDITOR DE VIDEOS VIRALES")
        print("=" * 50)
        print("Uso:")
        print("  Todos los clips: python editor_viral.py [carpeta_clips] [carpeta_salida]")
        print("  Un clip:         python editor_viral.py clips/clip_1.mp4 [carpeta_salida]")
        print("\nEjemplos:")
        print("  python editor_viral.py clips clips_editados")
        print("  python editor_viral.py clips/clip_1.mp4 clips_editados")
        print("=" * 50)
        sys.exit(1)
    
    clips_input = sys.argv[1]
    salida_dir = sys.argv[2] if len(sys.argv) > 2 else "clips_editados"
    
    input_path = Path(clips_input)
    if input_path.is_file():
        print(f"\nProcesando clip individual: {clips_input}")
        editor = VideoEditor(str(input_path.parent), salida_dir)
        resultado = editor.procesar_clip(input_path, 0)
        if resultado:
            print(f"\nClip procesado: {resultado}")
        else:
            print("\nError al procesar clip")
    else:
        editor = VideoEditor(clips_input, salida_dir)
        resultados = editor.procesar_todos()
        for r in resultados:
            print(f"  - {r}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Módulo de análisis de audio para detección de eventos virales.
Usa librosa para análisis de audio sin IA.
"""

import os
import json
import librosa
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class EventoAudio:
    """Representa un evento detectado en el audio."""
    timestamp: str
    evento: str
    intensidad: float


def cargar_audio(ruta: str) -> Tuple[np.ndarray, int]:
    """Carga un archivo de audio y retorna la señal y sample rate."""
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"Archivo no encontrado: {ruta}")
    
    print(f"  Cargando audio: {ruta}")
    y, sr = librosa.load(ruta, sr=None)
    duracion = len(y) / sr
    print(f"  Duración: {duracion:.1f}s | Sample rate: {sr}Hz")
    return y, sr


def calcular_rms(y: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula RMS (Root Mean Square) del audio."""
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    tiempos = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    return rms, tiempos


def detectar_eventos(rms: np.ndarray, tiempos: np.ndarray) -> List[EventoAudio]:
    """Detecta eventos en el audio basándose en RMS."""
    eventos = []
    
    media_rms = np.mean(rms)
    std_rms = np.std(rms)
    
    umbral_grito = media_rms + 2 * std_rms
    umbral_silencio = media_rms * 0.3
    umbral_cambio = std_rms
    
    print(f"  RMS media: {media_rms:.4f} | std: {std_rms:.4f}")
    
    i = 0
    while i < len(rms):
        frame_rms = rms[i]
        tiempo = tiempos[i]
        
        if frame_rms > umbral_grito:
            eventos.append(EventoAudio(
                timestamp=segundos_a_timestamp(tiempo),
                evento="grito",
                intensidad=float(frame_rms)
            ))
        
        elif frame_rms < umbral_silencio:
            eventos.append(EventoAudio(
                timestamp=segundos_a_timestamp(tiempo),
                evento="silencio",
                intensidad=float(1 - frame_rms / umbral_silencio) * 10
            ))
        
        if i > 0:
            cambio = abs(rms[i] - rms[i-1])
            if cambio > umbral_cambio and frame_rms > media_rms:
                eventos.append(EventoAudio(
                    timestamp=segundos_a_timestamp(tiempo),
                    evento="cambio_brusco",
                    intensidad=float(min(cambio / std_rms, 10))
                ))
        
        i += 1
    
    return eventos


def detectar_momentos_intensos(rms: np.ndarray, tiempos: np.ndarray, umbral: float = None) -> List[EventoAudio]:
    """Detecta momentos con energía alta sostenida."""
    eventos = []
    
    media_rms = np.mean(rms)
    std_rms = np.std(rms)
    
    if umbral is None:
        umbral = media_rms + 1.5 * std_rms
    
    ventana = 5
    i = 0
    while i <= len(rms) - ventana:
        chunk = rms[i:i + ventana]
        if np.mean(chunk) > umbral:
            tiempo_inicio = tiempos[i]
            eventos.append(EventoAudio(
                timestamp=segundos_a_timestamp(tiempo_inicio),
                evento="momento_intenso",
                intensidad=float(np.mean(chunk))
            ))
            i += ventana
        else:
            i += 1
    
    return eventos


def detectar_cambios_escena(y: np.ndarray, sr: int) -> List[EventoAudio]:
    """
    Detecta cambios abruptos de tema/escena usando spectral flux.
    Útil para encontrar puntos de corte naturales entre temas del video.
    """
    eventos = []
    
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            units='time',
            delta=0.5,
            wait=2
        )
        
        for t in onsets:
            eventos.append(EventoAudio(
                timestamp=segundos_a_timestamp(t),
                evento="cambio_escena",
                intensidad=5.0
            ))
    except Exception as e:
        print(f"  AVISO: Error detectando cambios de escena: {e}")
    
    return eventos


def limpiar_eventos(eventos: List[EventoAudio], separacion_min: int = 2) -> List[EventoAudio]:
    """Elimina eventos duplicados muy cercanos."""
    if not eventos:
        return []
    
    eventos_ordenados = sorted(eventos, key=lambda x: x.timestamp)
    eventos_limpios = [eventos_ordenados[0]]
    
    for actual in eventos_ordenados[1:]:
        ultimo = eventos_limpios[-1]
        diff = timestamp_diff(ultimo.timestamp, actual.timestamp)
        
        if diff >= separacion_min:
            eventos_limpios.append(actual)
        elif actual.intensidad > ultimo.intensidad:
            eventos_limpios[-1] = actual
    
    return eventos_limpios


def timestamp_diff(ts1: str, ts2: str) -> float:
    """Calcula diferencia en segundos entre dos timestamps."""
    s1 = timestamp_a_segundos(ts1)
    s2 = timestamp_a_segundos(ts2)
    return abs(s2 - s1)


def timestamp_a_segundos(timestamp: str) -> float:
    """Convierte HH:MM:SS a segundos."""
    partes = timestamp.split(':')
    while len(partes) < 3:
        partes.insert(0, '0')
    return int(partes[0]) * 3600 + int(partes[1]) * 60 + int(partes[2])


def segundos_a_timestamp(segundos: float) -> str:
    """Convierte segundos a HH:MM:SS."""
    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    segs = int(segundos % 60)
    return f"{horas:02d}:{minutos:02d}:{segs:02d}"


def normalizar_intensidad(eventos: List[EventoAudio]) -> List[EventoAudio]:
    """Normaliza intensidad a escala 0-10."""
    if not eventos:
        return eventos
    
    max_intensidad = max(e.intensidad for e in eventos)
    
    if max_intensidad > 0:
        for evento in eventos:
            intensidad_normalizada = (evento.intensidad / max_intensidad) * 10
            intensidad_normalizada = max(3.0, min(10.0, intensidad_normalizada))
            evento.intensidad = round(intensidad_normalizada, 1)
    
    return eventos


def filtrar_eventos(eventos: List[EventoAudio], intensidad_min: float = 3.0) -> List[EventoAudio]:
    """Elimina eventos con intensidad muy baja."""
    return [e for e in eventos if e.intensidad >= intensidad_min]


def eventos_a_dict(eventos: List[EventoAudio]) -> List[Dict]:
    """Convierte eventos a formato diccionario."""
    return [
        {
            "timestamp": e.timestamp,
            "evento": e.evento,
            "intensidad": e.intensidad
        }
        for e in eventos
    ]


def guardar_json(eventos: List[Dict], ruta: str = "audio.json") -> None:
    """Guarda los eventos en un archivo JSON."""
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(eventos, f, indent=2, ensure_ascii=False)
    print(f"  Guardado: {ruta}")


def detectar_momentos_virales_clustering(
    eventos: List[Dict], 
    n_clusters: int = 10
) -> List[Dict]:
    """
    Usa KMeans clustering para identificar los momentos más representativos
    y diversos del video, evitando clusters de momentos similares repetidos.
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        
        features = []
        eventos_idx = []
        
        for i, evento in enumerate(eventos):
            ts = timestamp_a_segundos(evento.get('timestamp', '00:00:00'))
            intensidad = evento.get('intensidad', 5.0)
            
            tipo_score = {
                'grito': 4,
                'momento_intenso': 3,
                'cambio_brusco': 2,
                'cambio_escena': 1,
                'silencio': 0
            }.get(evento.get('evento', 'silencio'), 0)
            
            features.append([ts, intensidad, tipo_score])
            eventos_idx.append(i)
        
        if len(features) < n_clusters:
            return sorted(eventos, key=lambda x: x.get('intensidad', 0), reverse=True)
        
        X = np.array(features)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        n_clusters_real = min(n_clusters, len(features))
        kmeans = KMeans(n_clusters=n_clusters_real, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        mejores_por_cluster = []
        for cluster_id in range(n_clusters_real):
            indices_cluster = [i for i, l in enumerate(labels) if l == cluster_id]
            if not indices_cluster:
                continue
            mejor_idx = max(indices_cluster, key=lambda i: eventos[i].get('intensidad', 0))
            mejores_por_cluster.append(eventos[mejor_idx])
        
        mejores_por_cluster.sort(key=lambda x: timestamp_a_segundos(x.get('timestamp', '00:00:00')))
        
        print(f"  Clustering: {n_clusters_real} clusters generados")
        return mejores_por_cluster
        
    except Exception as e:
        print(f"  Clustering falló: {e}")
        return sorted(eventos, key=lambda x: x.get('intensidad', 0), reverse=True)


def analizar_audio(audio_path: str = "audio.mp3", salida_path: str = "audio.json") -> List[Dict]:
    """
    Función principal: analiza un archivo de audio y genera audio.json.
    
    Args:
        audio_path: Ruta al archivo de audio (MP3)
        salida_path: Ruta donde guardar el JSON de salida
    
    Returns:
        Lista de diccionarios con los eventos detectados
    """
    print("=" * 40)
    print("  ANALIZADOR DE AUDIO")
    print("=" * 40)
    print("Analizando audio...\n")
    
    try:
        y, sr = cargar_audio(audio_path)
        rms, tiempos = calcular_rms(y, sr)
        
        print("\nDetectando eventos...")
        eventos = detectar_eventos(rms, tiempos)
        print(f"  Eventos detectados (gritos/silencios/cambios): {len(eventos)}")
        
        momentos_intensos = detectar_momentos_intensos(rms, tiempos)
        print(f"  Momentos intensos detectados: {len(momentos_intensos)}")
        
        cambios_escena = detectar_cambios_escena(y, sr)
        print(f"  Cambios de escena detectados: {len(cambios_escena)}")
        
        eventos.extend(momentos_intensos)
        eventos.extend(cambios_escena)
        
        print("\nLimpiando eventos...")
        eventos = limpiar_eventos(eventos, separacion_min=2)
        print(f"  Eventos después de limpieza: {len(eventos)}")
        
        print("\nNormalizando intensidad...")
        eventos = normalizar_intensidad(eventos)
        
        eventos = filtrar_eventos(eventos, intensidad_min=3.0)
        print(f"  Eventos después de filtrado: {len(eventos)}")
        
        if not eventos:
            print("\nNo se detectaron eventos relevantes")
            return []
        
        eventos_dict = eventos_a_dict(eventos)
        guardar_json(eventos_dict, salida_path)
        
        print("\nAgrupando momentos con clustering...")
        try:
            momentos_virales = detectar_momentos_virales_clustering(eventos_dict, n_clusters=10)
            guardar_json(momentos_virales, "momentos_virales.json")
            print(f"  Momentos virales seleccionados: {len(momentos_virales)}")
        except Exception as e:
            print(f"  Clustering falló: {e}")
        
        print(f"\n✓ Análisis completado")
        print(f"  Total eventos: {len(eventos_dict)}")
        print("=" * 40)
        
        return eventos_dict
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return []
    except Exception as e:
        print(f"\nError inesperado: {e}")
        return []


def main():
    """Ejemplo de uso."""
    import sys
    
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "audio.mp3"
    salida_path = sys.argv[2] if len(sys.argv) > 2 else "audio.json"
    
    analizar_audio(audio_path, salida_path)


if __name__ == "__main__":
    main()

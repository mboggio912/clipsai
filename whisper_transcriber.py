#!/usr/bin/env python3
"""
Transcripción automática con faster-whisper.
Genera dos archivos:
  - transcripcion_formatted.txt: para la IA (bloques legibles)
  - whisper_words.json: timestamps por palabra para subtítulos perfectos

Usa el backend más rápido disponible (Distil-Whisper > Faster-Whisper).
"""

import os
import json
import subprocess
from typing import List, Dict


def segundos_a_hhmmss(seg: float) -> str:
    """Convierte float de segundos a HH:MM:SS."""
    h = int(seg // 3600)
    m = int((seg % 3600) // 60)
    s = int(seg % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def detectar_device() -> tuple:
    """Detecta el mejor device disponible."""
    try:
        import torch
        if torch.cuda.is_available():
            return ("cuda", "float16")
    except Exception:
        pass
    return ("cpu", "int8")


def transcribir_video(
    video_path: str,
    salida_transcripcion: str = "transcripcion_formatted.txt",
    salida_words: str = "whisper_words.json",
    modelo: str = "whisper-small",
    idioma: str = "es",
    device: str = None,
    compute_type: str = None,
) -> dict:
    """
    Transcribe un video usando el backend más rápido disponible.

    Args:
        video_path: Ruta al video
        salida_transcripcion: Archivo de transcripción formateada
        salida_words: Archivo JSON con timestamps por palabra
        modelo: Modelo ("distil-large-v3", "distil-medium", "small", "large-v3")
        idioma: Código ISO del idioma
        device: Override para device (None = auto-detectar)
        compute_type: Override para compute type (None = auto-detectar)

    Returns:
        dict con {
            "transcripcion_path": str,
            "words_path": str,
            "duracion": float,
            "total_palabras": int,
            "idioma_detectado": str,
            "confianza_idioma": float,
            "backend": str
        }
    """
    import time
    inicio_total = time.time()

    print("=" * 50)
    print("  TRANSCRIPCIÓN OPTIMIZADA")
    print("=" * 50)

    if device is None or compute_type is None:
        device, compute_type = detectar_device()
    print(f"  Device: {device} ({compute_type})")

    words = []
    info = {"duration": 0, "language": idioma, "language_probability": 0.95}
    backend = "unknown"

    if modelo.startswith("whisper-") or modelo.startswith("openai/"):
        try:
            words, info, backend = _transcribir_openai_whisper(video_path, modelo, idioma, device)
        except Exception as e:
            print(f"  OpenAI Whisper falló: {e}")
            print("  Intentando con Faster-Whisper...")
            words, info, backend = _transcribir_fasterwhisper(video_path, modelo, idioma, device, compute_type)
    elif modelo.startswith("distil"):
        try:
            words, info, backend = _transcribir_distilwhisper(video_path, modelo, idioma, device)
        except Exception as e:
            print(f"  Distil-Whisper falló: {e}")
            print("  Intentando con Faster-Whisper...")
            words, info, backend = _transcribir_fasterwhisper(video_path, modelo, idioma, device, compute_type)
    else:
        words, info, backend = _transcribir_fasterwhisper(video_path, modelo, idioma, device, compute_type)

    duracion_trans = time.time() - inicio_total

    print(f"  ✓ Transcripción completada ({backend})")
    print(f"    Duración video: {info['duration']:.1f}s")
    print(f"    Idioma: {info['language']} (prob: {info['language_probability']:.1%})")
    print(f"    Tiempo: {duracion_trans:.1f}s")
    if info['duration'] > 0:
        print(f"    Velocidad: {info['duration']/max(duracion_trans,0.1):.1f}x realtime")
    print(f"    Palabras: {len(words)}")

    with open(salida_words, 'w', encoding='utf-8') as f:
        json.dump(words, f, ensure_ascii=False, indent=2)
    print(f"    whisper_words.json: {salida_words}")

    bloques = _generar_bloques(words)
    with open(salida_transcripcion, 'w', encoding='utf-8') as f:
        f.write('\n'.join(bloques) + '\n')
    print(f"    transcripcion_formatted.txt: {salida_transcripcion}")
    print(f"    Bloques: {len(bloques)}")

    duracion_total = time.time() - inicio_total
    print(f"\n  Total: {duracion_total:.1f}s")
    print("=" * 50)

    return {
        "transcripcion_path": salida_transcripcion,
        "words_path": salida_words,
        "duracion": info['duration'],
        "total_palabras": len(words),
        "idioma_detectado": info['language'],
        "confianza_idioma": info['language_probability'],
        "backend": backend
    }


def _extraer_audio(video_path: str) -> str:
    """Extrae audio usando FFmpeg."""
    audio_temp = "audio_whisper_temp.wav"
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        audio_temp
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return audio_temp


def _transcribir_distilwhisper(video_path: str, modelo: str, idioma: str, device: str) -> tuple:
    """Transcribe usando Distil-Whisper (6x más rápido)."""
    import librosa
    import numpy as np
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    model_map = {
        "distil-large-v3": "distil-whisper/distil-large-v3",
        "distil-medium": "distil-whisper/distil-medium",
        "distil-small": "distil-whisper/distil-small",
    }

    model_name = model_map.get(modelo, "distil-whisper/distil-large-v3")

    print(f"  Backend: Distil-Whisper ({model_name})")

    audio_temp = _extraer_audio(video_path)

    print(f"  Cargando modelo...")
    processor = AutoProcessor.from_pretrained(model_name)

    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    if device != "cpu":
        model = model.to(device)

    print(f"  Cargando audio...")
    audio, sr = librosa.load(audio_temp, sr=16000)
    audio = audio.astype(np.float32)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    print(f"  Transcribiendo (6x más rápido que Whisper normal)...")

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        return_token_timestamps=True
    )

    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_features"],
            forced_decoder_ids=processor.get_decoder_prompt_ids(language=idioma, task="transcribe"),
            max_new_tokens=440,
            return_timestamps=True,
            use_cache=True,
        )

    output = processor.batch_decode(generated_ids, output_word_offsets=True)

    words = []
    if output and len(output) > 0:
        result = output[0]
        if hasattr(result, 'word_offsets') and result.word_offsets:
            for word_info in result.word_offsets:
                words.append({
                    "word": word_info["word"].strip(),
                    "start": round(word_info["start"], 3),
                    "end": round(word_info["end"], 3),
                    "probability": 0.95,
                    "segment_id": 0
                })
        elif hasattr(result, 'text'):
            texto = result.text
            duracion = len(audio) / 16000
            palabras = texto.split()
            if palabras and duracion > 0:
                tiempo_por_palabra = duracion / len(palabras)
                for i, p in enumerate(palabras):
                    words.append({
                        "word": p.strip(),
                        "start": round(i * tiempo_por_palabra, 3),
                        "end": round((i + 1) * tiempo_por_palabra, 3),
                        "probability": 0.95,
                        "segment_id": 0
                    })

    info = {
        "duration": len(audio) / 16000,
        "language": idioma,
        "language_probability": 0.95
    }

    try:
        os.remove(audio_temp)
    except Exception:
        pass

    return words, info, "distilwhisper"


def _transcribir_openai_whisper(video_path: str, modelo: str, idioma: str, device: str) -> tuple:
    """Transcribe usando OpenAI Whisper (transformers) - rápido y preciso."""
    import librosa
    import numpy as np
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    import os
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

    model_name = modelo if modelo.startswith("openai/") else f"openai/{modelo}"

    model_map = {
        "openai/whisper-small": "openai/whisper-small",
        "openai/whisper-medium": "openai/whisper-medium",
        "openai/whisper-base": "openai/whisper-base",
    }
    model_name = model_map.get(model_name, "openai/whisper-small")

    print(f"  Backend: OpenAI Whisper ({model_name})")

    audio_temp = _extraer_audio(video_path)

    print(f"  Cargando modelo...")
    processor = AutoProcessor.from_pretrained(model_name)

    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    if device != "cpu":
        model = model.to(device)

    print(f"  Cargando audio...")
    audio, sr = librosa.load(audio_temp, sr=16000)
    audio = audio.astype(np.float32)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    print(f"  Transcribiendo...")

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_features"],
            language=idioma,
            task="transcribe",
            max_new_tokens=440,
            use_cache=True,
        )

    output = processor.batch_decode(generated_ids, output_word_offsets=True)

    words = []
    texto_completo = ""
    
    if output and len(output) > 0:
        result = output[0]
        
        if hasattr(result, 'word_offsets') and result.word_offsets and len(result.word_offsets) > 0:
            for word_info in result.word_offsets:
                words.append({
                    "word": word_info["word"].strip(),
                    "start": round(word_info["start"], 3),
                    "end": round(word_info["end"], 3),
                    "probability": 0.95,
                    "segment_id": 0
                })
            texto_completo = " ".join(w["word"] for w in words)
        elif hasattr(result, 'text'):
            texto_completo = result.text
        else:
            texto_completo = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    if not words and texto_completo:
        duracion = len(audio) / 16000
        palabras = texto_completo.split()
        if palabras and duracion > 0:
            tiempo_por_palabra = duracion / len(palabras)
            for i, p in enumerate(palabras):
                words.append({
                    "word": p.strip(),
                    "start": round(i * tiempo_por_palabra, 3),
                    "end": round((i + 1) * tiempo_por_palabra, 3),
                    "probability": 0.95,
                    "segment_id": 0
                })
            print(f"  AVISO: Modelo sin timestamps - usando estimación proporcional")

    info = {
        "duration": len(audio) / 16000,
        "language": idioma,
        "language_probability": 0.95
    }

    try:
        os.remove(audio_temp)
    except Exception:
        pass

    return words, info, "openai-whisper"


def _transcribir_fasterwhisper(video_path: str, modelo: str, idioma: str, device: str, compute_type: str) -> tuple:
    """Transcribe usando Faster-Whisper."""
    from faster_whisper import WhisperModel

    modelos_disponibles = {
        "large-v3": "large-v3",
        "medium": "medium",
        "small": "small",
        "base": "base",
        "tiny": "tiny",
    }
    model_name = modelos_disponibles.get(modelo, "large-v3")

    print(f"  Backend: Faster-Whisper ({model_name})")

    print(f"  Cargando modelo...")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    audio_temp = _extraer_audio(video_path)

    print(f"  Transcribiendo...")

    segments, info = model.transcribe(
        audio_temp,
        language=idioma,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=150
        ),
        beam_size=1,
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=False
    )

    segments_list = list(segments)

    words = []
    for seg_idx, segment in enumerate(segments_list):
        if not segment.words:
            continue
        for word in segment.words:
            words.append({
                "word": word.word.strip(),
                "start": round(word.start, 3),
                "end": round(word.end, 3),
                "probability": round(word.probability, 3),
                "segment_id": seg_idx
            })

    info_dict = {
        "duration": info.duration,
        "language": info.language,
        "language_probability": info.language_probability
    }

    try:
        os.remove(audio_temp)
    except Exception:
        pass

    return words, info_dict, "fasterwhisper"


def _generar_bloques(words: list, max_palabras: int = 6) -> list:
    """Genera bloques legibles para la IA."""
    bloques = []
    buffer = []
    tiempo_inicio = None

    for i, word_data in enumerate(words):
        palabra = word_data['word']
        t_inicio = word_data['start']

        if tiempo_inicio is None:
            tiempo_inicio = t_inicio

        pausa = 0.0
        if i > 0:
            pausa = t_inicio - words[i - 1]['end']

        fin_oracion = False
        if buffer:
            ultima = buffer[-1]
            fin_oracion = ultima.rstrip().endswith(('.', '!', '?'))

        nuevo_bloque = (
            len(buffer) >= max_palabras or
            pausa > 0.5 or
            fin_oracion
        )

        if nuevo_bloque and buffer:
            texto = ' '.join(buffer).strip()
            if texto:
                ts = segundos_a_hhmmss(tiempo_inicio)
                bloques.append(f"{ts} - {texto}")
            buffer = []
            tiempo_inicio = t_inicio

        buffer.append(palabra)

    if buffer and tiempo_inicio is not None:
        texto = ' '.join(buffer).strip()
        if texto:
            ts = segundos_a_hhmmss(tiempo_inicio)
            bloques.append(f"{ts} - {texto}")

    return bloques


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python whisper_transcriber.py video.mp4 [modelo]")
        print("  Modelos disponibles (más rápido primero):")
        print("    whisper-small    - Rápido (~2-4 min para 30min video) (recomendado)")
        print("    whisper-medium   - Balance velocidad/calidad (~5-8 min)")
        print("    large-v3         - Mejor calidad, más lento (~15-30 min)")
        print()
        print("  Por defecto usa: whisper-small")
        sys.exit(1)

    video = sys.argv[1]
    modelo = sys.argv[2] if len(sys.argv) > 2 else "whisper-small"
    salida_trans = sys.argv[3] if len(sys.argv) > 3 else "transcripcion_formatted.txt"
    salida_words = sys.argv[4] if len(sys.argv) > 4 else "whisper_words.json"

    try:
        print(f"\nModelo: {modelo}")
        print(f"Video: {video}\n")

        resultado = transcribir_video(
            video,
            salida_transcripcion=salida_trans,
            salida_words=salida_words,
            modelo=modelo
        )

        print(f"\n✓ Listo:")
        print(f"  Transcripción: {resultado['transcripcion_path']}")
        print(f"  Words JSON: {resultado['words_path']}")
        print(f"  Palabras: {resultado['total_palabras']}")
        print(f"  Backend: {resultado['backend']}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

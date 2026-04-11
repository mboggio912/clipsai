# Generador de Clips Virales

Programa Python que automatiza la generación de clips a partir de una transcripción usando IA.

## Requisitos

- Python 3.7+
- FFmpeg instalado
- API Key de OpenRouter

## Instalación

```bash
pip install requests
```

Instala FFmpeg:
- **Linux**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Descarga desde https://ffmpeg.org/download.html

## Configuración

Configura tu API key de OpenRouter:

```bash
export OPENROUTER_API_KEY="tu-api-key-aqui"
```

O alternativamente, modificada `main.py` para cargarla desde un archivo `.env`.

## Uso

```bash
python main.py video.mp4 transcripcion.txt
```

Los clips se guardarán en la carpeta `clips/`.

## Formato de transcripción

El archivo de transcripción debe ser texto plano. Puede incluir timestamps opcionales:

```
[00:01:30] Este es el primer momento destacado
[00:02:45] Aquí pasó algo gracioso
```

## Ejemplo

```bash
python main.py mi_video.mp4 ejemplo_transcripcion.txt
```

## Salida

```
==================================================
  GENERADOR DE CLIPS VIRALES
==================================================

[1/5] Leyendo transcripción...
      Transcripción leída: 1500 caracteres
[2/5] Analizando transcripción con IA...
      IA detectó 3 momentos potenciales
[3/5] Validando clips...
      Clip 1: 00:03:00 - 00:04:00 (60s) - Momento gracioso
      Clip 2: 00:07:00 - 00:08:00 (60s) - Secreto revelado
      Total clips válidos: 2
[4/5] Generando clips...
      Generando clip 1: 00:03:00 - 00:04:00
      ✓ Clip 1 guardado: clips/clip_1.mp4
[5/5] Proceso completado
==================================================
  CLIPS GENERADOS: 2
  UBICACIÓN: /ruta/proyecto/clips
==================================================
```
# clipsai

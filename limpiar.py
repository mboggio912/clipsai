#!/usr/bin/env python3
"""
Limpia todos los archivos generados/temporales del proyecto.
Deja solo el código fuente.
"""

import os
import shutil
from pathlib import Path

ARCHIVOS_A_BORRAR = [
    "audio.mp3",
    "audio.wav",
    "audio_trans_temp.wav",
    "audio.json",
    "momentos_virales.json",
    "transcripcion.txt",
    "transcripcion_formatted.txt",
    "vidstab_transforms.trf",
]

CARPETAS_A_BORRAR = [
    "clips",
    "clips_editados",
    "__pycache__",
]

EXTENSIONES_PY_CACHE = ["*.pyc", "*.pyo", "*.pyd"]


def limpiar_proyecto():
    print("=" * 50)
    print("  LIMPIANDO PROYECTO")
    print("=" * 50)
    
    proyecto = Path(".")
    archivos_borrados = 0
    carpetas_borradas = 0
    
    # Borrar archivos individuales
    print("\nArchivos generados:")
    for nombre in ARCHIVOS_A_BORRAR:
        p = Path(nombre)
        if p.exists():
            p.unlink()
            print(f"  ✓ {nombre}")
            archivos_borrados += 1
        else:
            print(f"  - {nombre} (no existe)")
    
    # Borrar carpetas
    print("\nCarpetas generadas:")
    for nombre in CARPETAS_A_BORRAR:
        p = Path(nombre)
        if p.exists() and p.is_dir():
            shutil.rmtree(p)
            print(f"  ✓ {nombre}/")
            carpetas_borradas += 1
        else:
            print(f"  - {nombre}/ (no existe)")
    
    # Borrar __pycache__ en subcarpetas
    print("\nCache de Python:")
    for p in proyecto.rglob("__pycache__"):
        shutil.rmtree(p)
        print(f"  ✓ {p}")
        archivos_borrados += 1
    
    # Borrar archivos .pyc
    print("\nArchivos .pyc:")
    for ext in EXTENSIONES_PY_CACHE:
        for p in proyecto.rglob(ext):
            p.unlink()
            print(f"  ✓ {p.name}")
            archivos_borrados += 1
    
    print("\n" + "=" * 50)
    print(f"  TOTAL: {archivos_borrados} archivos, {carpetas_borradas} carpetas borrados")
    print("=" * 50)


if __name__ == "__main__":
    limpiar_proyecto()
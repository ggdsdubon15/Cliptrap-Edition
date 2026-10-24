from fastapi import FastAPI, UploadFile, File
import shutil
import os
from faster_whisper import WhisperModel

app = FastAPI(title="Cliptrap AI Backend", version="1.0")

# Cargamos el modelo de Whisper (puedes usar 'tiny', 'base', 'small', 'medium' o 'large')
# 'base' o 'small' son excelentes para pruebas rápidas en la nube
model_size = "base"
print("Cargando modelo de Whisper...")
model = WhisperModel(model_size, device="cpu", compute_type="int8")
print("¡Modelo cargado con éxito!")

@app.post("/transcribe/")
async def transcribe_video(file: UploadFile = File(...)):
    # Guardar el video temporalmente
    temp_video_path = f"temp_{file.filename}"
    with open(temp_video_path, "buffer.wb" if hasattr(shutil, "wb") else "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    print(f"Procesando video: {file.filename}")
    
    # Transcribir y obtener marcas de tiempo por segmento/palabra
    segments, info = model.transcribe(temp_video_path, beam_size=5)
    
    subtitles = []
    for segment in segments:
        subtitles.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })
    
    # Limpiar archivo temporal
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
        
    return {
        "language": info.language,
        "duration": info.duration,
        "subtitles": subtitles
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

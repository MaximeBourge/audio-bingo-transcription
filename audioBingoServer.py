from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from gradio_client import Client, handle_file
import os
import uvicorn

app = FastAPI()

# Dossier temporaire
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Client HuggingFace
client = Client("GigaMaxime/audio-bingo-whisper")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    print("[SERVER] Requête reçue sur /transcribe")

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    print(f"[SERVER] Fichier reçu : {file_path}")
    print(f"[SERVER] Taille du fichier : {os.path.getsize(file_path)} octets")

    try:
        print("[SERVER] Envoi du fichier à HuggingFace...")
        result = client.predict(
            file=handle_file(file_path),
            api_name="/predict"
        )
        print("[SERVER] Réponse HuggingFace reçue")

        return {"transcription": result}
    except Exception as e:
        print(f"[SERVER] Erreur : {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    print("[SERVER] Démarrage du serveur FastAPI sur http://0.0.0.0:5000")
    uvicorn.run(app, host="0.0.0.0", port=5000, reload=True)

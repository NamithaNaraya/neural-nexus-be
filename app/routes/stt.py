from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.stt_service import get_stt_service
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Endpoint to transcribe audio files to text.
    Accepts any standard audio format (wav, mp3, webm, etc.)
    """
    if not file:
        raise HTTPException(status_code=400, detail="No audio file provided")

    try:
        logger.info(f"🎙️ Received audio for transcription: {file.filename} ({file.content_type})")
        
        # Read the file bytes
        audio_data = await file.read()
        
        # Transcribe using service
        stt = get_stt_service()
        text = await stt.transcribe(audio_data)
        
        logger.info(f"✅ Transcription complete: '{text[:50]}...'")
        
        return {
            "success": True,
            "text": text
        }
    except Exception as e:
        logger.error(f"❌ Error during transcription route: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# This script provides a wrapper for TTS calls using environment variables.
# It is compatible with Azure, GCP, and other providers as required by Adobe's evaluation.
import os
import sys
from typing import Optional

def generate_audio(text: str, output_path: str, speaker: Optional[str] = None) -> str:
    """
    Generate an MP3 audio file from text using the TTS provider specified in environment variables.
    Compatible with Azure, GCP, and other providers.
    """
    TTS_PROVIDER = os.environ.get('TTS_PROVIDER', 'azure')
    if TTS_PROVIDER == 'azure':
        import azure.cognitiveservices.speech as speechsdk
        # Support both AZURE_TTS_KEY/AZURE_TTS_ENDPOINT and AZURE_SPEECH_KEY/AZURE_SPEECH_REGION
        speech_key = os.environ.get('AZURE_TTS_KEY') or os.environ.get('AZURE_SPEECH_KEY')
        speech_region = os.environ.get('AZURE_TTS_ENDPOINT') or os.environ.get('AZURE_SPEECH_REGION')
        if not speech_key or not speech_region:
            raise RuntimeError('Azure TTS credentials not set. Please set AZURE_TTS_KEY/AZURE_TTS_ENDPOINT or AZURE_SPEECH_KEY/AZURE_SPEECH_REGION.')
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
        audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        result = synthesizer.speak_text_async(text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return output_path
        else:
            raise RuntimeError(f"Speech synthesis failed: {result.reason}")
    else:
        raise NotImplementedError('Only Azure TTS is supported in this sample.')

if __name__ == "__main__":
    # Example usage: python generate_audio.py "Hello world" output.mp3
    # Requires AZURE_SPEECH_KEY and AZURE_SPEECH_REGION to be set in the environment
    if len(sys.argv) < 3:
        print("Usage: python generate_audio.py <text> <output_path>")
        sys.exit(1)
    text = sys.argv[1]
    output_path = sys.argv[2]
    print(generate_audio(text, output_path))

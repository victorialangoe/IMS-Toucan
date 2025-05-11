from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface

tts = ToucanTTSInterface(tts_model_path='../IMS-Toucan/checkpoints/trondelag/best.pt')

tts.set_language("nob")

speaker_reference_path = "../audios/speaker_references/trondelag_ref.wav"

tts.set_utterance_embedding(speaker_reference_path)

text = "På fritia like æ å spæll fottball"

tts.read_to_file([text], file_location="teste_litt.wav")

import speakerlab.models.en_kd.wavlm_ecapa_tdnn as wavlm_ecapa_tdnn

def get_speaker_model(model_name: str):
    if model_name.startswith("ECAPA_TDNN_SMALL"):
        return getattr(wavlm_ecapa_tdnn, model_name)
    else: # model name error !!!
        print(model_name + " not found !!!")
        exit(1)
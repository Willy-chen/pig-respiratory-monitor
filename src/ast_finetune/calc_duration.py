import os
import glob
import librosa

def get_dir_duration(path):
    duration = 0
    files = glob.glob(os.path.join(path, "*.wav"))
    for f in files:
        try:
            duration += librosa.get_duration(path=f)
        except:
            pass
    return duration / 60.0  # Return in minutes

_AUDIO_ROOT = os.environ.get("PIG_AUDIO_ROOT", "../../audio")
normal_path   = os.path.join(_AUDIO_ROOT, "diagnostic_labelled", "normal")
abnormal_path = os.path.join(_AUDIO_ROOT, "diagnostic_labelled", "abnormal")

norm_dur = get_dir_duration(normal_path)
abnorm_dur = get_dir_duration(abnormal_path)

print(f"Normal Duration: {norm_dur:.2f} minutes")
print(f"Abnormal Duration: {abnorm_dur:.2f} minutes")
print(f"Total Duration: {norm_dur + abnorm_dur:.2f} minutes")

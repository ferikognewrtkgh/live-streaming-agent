from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
LIVE2D_MODELS_ROOT = PROJECT_ROOT / "live2d-models"
BACKGROUND_ROOT = PROJECT_ROOT / "backgrounds"
AVATAR_ROOT = PROJECT_ROOT / "avatars"
MODEL_DICT_PATH = PROJECT_ROOT / "model_dict.json"
SLEEP_VOICE_ROOT = PROJECT_ROOT / "resource" / "sleep_voice"
ASR_CORRECTIONS_PATH = PROJECT_ROOT / "resource" / "asr_corrections.txt"

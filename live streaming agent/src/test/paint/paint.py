from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
import base64
import mimetypes
import os
import time

load_dotenv()

MODEL_CONFIGS = {
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key_envs": ["ZHIPUAI_API_KEY"],
        "model": "glm-5v-turbo",
        "extra_body": {"thinking": {"type": "disabled"}}
    },
    "ark": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "api_key_envs": ["ARK_API_KEY"],
        "model": "doubao-seed-2-0-code-preview-260215",
        "extra_body": {"thinking": {"type": "disabled"}},
    },
}

provider_name = 'glm'
image_path = './1.jpg'
model_config = MODEL_CONFIGS[provider_name]
api_key = next(
    (os.getenv(api_key_env) for api_key_env in model_config["api_key_envs"] if os.getenv(api_key_env)),
    None,
)

if not api_key:
    raise RuntimeError(f"请先配置 API Key：{', '.join(model_config['api_key_envs'])}")

model_name = os.getenv("MODEL_NAME", model_config["model"])

client = OpenAI(
    base_url=os.getenv("BASE_URL", model_config["base_url"]),
    api_key=api_key,
)

image = Path(image_path).expanduser().resolve()
mime_type = mimetypes.guess_type(image.name)[0] or "image/jpeg"
encoded = base64.b64encode(image.read_bytes()).decode("ascii")
image_base64 = f"data:{mime_type};base64,{encoded}"

system_prompt = (
    "请使用 Python turtle 库按照要求绘制一幅图片，"
    "只写出代码，不要其它内容，必须在画图前调用 turtle.tracer(False)。"
)
user_prompt = os.getenv("PAINT_PROMPT", "根据图片画一幅简化图")

request_params = {
    "model": model_name,
    "messages": [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_base64}},
                {"type": "text", "text": user_prompt},
            ],
        },
    ],
    "temperature": 0,
    "stream": True,
}

if model_config.get("extra_body"):
    request_params["extra_body"] = model_config["extra_body"]

start_time = time.perf_counter()
stream = client.chat.completions.create(**request_params)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="")

print()
print(time.perf_counter() - start_time)

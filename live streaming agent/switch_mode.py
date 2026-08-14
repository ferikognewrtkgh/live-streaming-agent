"""
模式切换命令行工具
用法: uv run python switch_mode.py <模式>

模式:
  sleep    - 休眠（静止模式）
  co_host  - 助播模式
  barrage  - 弹幕模式
  punish   - 罚站模式
"""

import asyncio
import json
import sys
import websockets

URI = "ws://127.0.0.1:12393/client-ws"


async def switch(mode: str):
    async with websockets.connect(URI) as ws:
        # 跳过初始消息
        for _ in range(5):
            try:
                await asyncio.wait_for(ws.recv(), timeout=2)
            except asyncio.TimeoutError:
                break

        await ws.send(json.dumps({"type": "console-message", "action": mode}))
        print(f">>> 已发送: {mode}")

        try:
            resp = await asyncio.wait_for(ws.recv(), timeout=3)
            data = json.loads(resp)
            if data.get("type") == "mode-changed":
                d = data.get("detail", {})
                print(f"<<< 切换成功: {d.get('old_mode')} -> {d.get('new_mode')} (action={d.get('action')})")
            else:
                print(f"<<< {resp[:200]}")
        except asyncio.TimeoutError:
            print("<<< (无响应)")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in {"sleep", "co_host", "barrage", "punish"}:
        print("用法: uv run python switch_mode.py <sleep|co_host|barrage|punish>")
        sys.exit(1)
    asyncio.run(switch(sys.argv[1]))

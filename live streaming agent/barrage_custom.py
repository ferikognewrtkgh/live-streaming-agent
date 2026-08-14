"""
弹幕自定义筛选/排序命令行工具
仿照 switch_mode.py 的写法, 通过 /client-ws 发 console-message.

用法:
  # 设置: wealth>=15, fan_badge 关闭, diamond_rank 关闭
  uv run python barrage_custom.py set wealth:15 fan_badge:off diamond_rank:off

  # 设置: wealth>=15 优先级0, diamond_rank>=1000 优先级1, fan_badge 关闭
  uv run python barrage_custom.py set wealth:15:0 diamond_rank:1000:1 fan_badge:off

  # 一次性关闭所有变量 → 回退到原有逻辑
  uv run python barrage_custom.py off

  # 查询当前配置
  uv run python barrage_custom.py status

  # 重置钻石累计 (切播间时用)
  uv run python barrage_custom.py reset-diamond

  # 查看本场钻石数累计排行 (全部)
  uv run python barrage_custom.py top

  # 只看前 10 名
  uv run python barrage_custom.py top 10

  # 查看 metrics (接收速率/队列长度/年龄分布)
  uv run python barrage_custom.py metrics

  # 读所有可热更新字段当前值
  uv run python barrage_custom.py config-get

  # 热更新配置: 改节奏 + 改关键词
  uv run python barrage_custom.py config-set consume_interval=4 \
      'keyword_list=["主播","老板","姐姐"]' welcome_enabled=true

  # 一键开启所有事件回复 (欢迎/关注)
  uv run python barrage_custom.py config-set welcome_enabled=true follow_enabled=true

参数语法:
  <name>:<threshold>          → 启用, 阈值=threshold, 优先级保持原值
  <name>:<threshold>:<priority> → 启用, 阈值=threshold, 优先级=priority
  <name>:off                  → 关闭该变量
  name 可选: wealth | fan_badge | diamond_rank
"""

import asyncio
import json
import sys

import websockets

URI = "ws://127.0.0.1:12393/client-ws"
VALID_NAMES = {"wealth", "fan_badge", "diamond_rank"}


def parse_var_spec(spec: str) -> dict:
    """解析 'wealth:15' 或 'wealth:15:0' 或 'wealth:off'"""
    parts = spec.split(":")
    name = parts[0]
    if name not in VALID_NAMES:
        raise ValueError(
            f"未知变量名: {name} (合法: {sorted(VALID_NAMES)})"
        )
    if len(parts) >= 2 and parts[1].lower() == "off":
        return {"name": name, "enabled": False}
    entry = {"name": name, "enabled": True}
    if len(parts) >= 2 and parts[1]:
        entry["threshold"] = int(parts[1])
    if len(parts) >= 3 and parts[2]:
        entry["priority"] = int(parts[2])
    return entry


async def _drain_initial(ws, max_msgs: int = 5, timeout: float = 2.0):
    """跳过服务端启动时发的初始化消息"""
    for _ in range(max_msgs):
        try:
            await asyncio.wait_for(ws.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            return


async def send_and_recv(payload: dict, expected_type: str):
    async with websockets.connect(URI) as ws:
        await _drain_initial(ws)
        await ws.send(json.dumps(payload, ensure_ascii=False))
        print(f">>> 已发送: {json.dumps(payload, ensure_ascii=False)}")

        # 等指定 type 的响应, 跳过无关消息
        end = asyncio.get_event_loop().time() + 5.0
        while asyncio.get_event_loop().time() < end:
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=3)
            except asyncio.TimeoutError:
                print("<<< (无响应)")
                return
            try:
                data = json.loads(resp)
            except Exception:
                continue
            if data.get("type") == expected_type:
                print("<<< 响应:")
                print(json.dumps(data, indent=2, ensure_ascii=False))
                return


async def cmd_set(specs: list):
    variables = [parse_var_spec(s) for s in specs]
    await send_and_recv(
        {
            "type": "console-message",
            "action": "barrage-custom-config",
            "variables": variables,
        },
        expected_type="barrage-custom-config",
    )


async def cmd_off():
    """一键全关 → 回退到原有逻辑"""
    variables = [
        {"name": "wealth", "enabled": False},
        {"name": "fan_badge", "enabled": False},
        {"name": "diamond_rank", "enabled": False},
    ]
    await send_and_recv(
        {
            "type": "console-message",
            "action": "barrage-custom-config",
            "variables": variables,
        },
        expected_type="barrage-custom-config",
    )


async def cmd_status():
    await send_and_recv(
        {"type": "console-message", "action": "barrage-custom-status"},
        expected_type="barrage-custom-status",
    )


async def cmd_reset_diamond():
    await send_and_recv(
        {"type": "console-message", "action": "barrage-reset-diamond"},
        expected_type="barrage-reset-diamond",
    )


async def cmd_metrics():
    """查询 metrics"""
    payload = {"type": "console-message", "action": "barrage-metrics"}
    async with websockets.connect(URI) as ws:
        await _drain_initial(ws)
        await ws.send(json.dumps(payload))
        print(">>> 查询 metrics")
        end = asyncio.get_event_loop().time() + 5.0
        while asyncio.get_event_loop().time() < end:
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=3)
            except asyncio.TimeoutError:
                print("<<< (无响应)")
                return
            try:
                data = json.loads(resp)
            except Exception:
                continue
            if data.get("type") != "barrage-metrics":
                continue
            if data.get("data") is None:
                print(f"<<< 错误: {data.get('error')}")
                return
            d = data["data"]
            print(f"<<< 窗口: 最近 {d['window_s']:.0f} 秒")
            r = d["rates_per_s"]
            print(
                f"    速率/秒: recv={r['recv']:.2f}  "
                f"enqueue={r['enqueue']:.2f}  "
                f"consume={r['consume']:.2f}  "
                f"filter_drop={r['filter_drop']:.2f}  "
                f"combo_merged={r['combo_merged']:.2f}"
            )
            q = d["queues"]
            print(
                f"    队列长度: 弹幕高={q['barrage_high']} 普={q['barrage_normal']} "
                f"关={q['barrage_keyword']} 礼物={q['gift']} "
                f"自定关={q['custom_keyword']} 自定普={q['custom_normal']}"
            )
            print(
                f"    缓冲: 礼物连击={q['gift_combo_buffer']} "
                f"待欢迎={q['pending_joins']} "
                f"待关注={q['pending_follows']}"
            )
            age = d["queue_age_seconds"]
            if age["custom_p50"] is not None:
                print(
                    f"    自定队列年龄(秒): p50={age['custom_p50']} "
                    f"p95={age['custom_p95']} max={age['custom_max']}"
                )
            c = d["counters"]
            print(
                f"    累计: 总收={c['total_received']} "
                f"重连={c['reconnect_count']} "
                f"点赞={c['like_count']} "
                f"钻石淘汰={c['diamond_evicted_total']} "
                f"自定丢={c['custom_dropped_total']} "
                f"自定过={c['custom_filtered_total']}"
            )
            if d.get("room_stats"):
                rs = d["room_stats"]
                print(
                    f"    直播间: 在线={rs.get('member_count')} "
                    f"累计观众={rs.get('total_user_count')}"
                )
            return


async def cmd_config_get():
    """读所有可热更新字段当前值"""
    payload = {"type": "console-message", "action": "barrage-runtime-config"}
    async with websockets.connect(URI) as ws:
        await _drain_initial(ws)
        await ws.send(json.dumps(payload))
        end = asyncio.get_event_loop().time() + 5.0
        while asyncio.get_event_loop().time() < end:
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=3)
            except asyncio.TimeoutError:
                print("<<< (无响应)")
                return
            try:
                data = json.loads(resp)
            except Exception:
                continue
            if data.get("type") != "barrage-runtime-config":
                continue
            if data.get("data") is None:
                print(f"<<< 错误: {data.get('error')}")
                return
            print("<<< 当前可热更新字段:")
            for k, v in data["data"].items():
                print(f"    {k:40} = {v}")
            return


async def cmd_config_set(kvs: list):
    """热更新一组字段, 支持 k=v 简单字符串和 k=<json> 复杂值"""
    patch: dict = {}
    for kv in kvs:
        if "=" not in kv:
            print(f"忽略无效参数 (缺 '=' ): {kv}")
            continue
        k, v = kv.split("=", 1)
        k = k.strip()
        v = v.strip()
        # 尝试 JSON 解析 (支持 true/false/数字/列表), 否则当字符串
        try:
            parsed = json.loads(v)
        except Exception:
            parsed = v
        patch[k] = parsed

    if not patch:
        print("没有可应用的 patch")
        return

    payload = {
        "type": "console-message",
        "action": "barrage-runtime-config",
        "patch": patch,
    }
    print(f">>> 热更新 patch: {json.dumps(patch, ensure_ascii=False)}")
    await send_and_recv(payload, expected_type="barrage-runtime-config")


async def cmd_top(top_n: int = 0):
    """查询本场钻石数累计排行"""
    payload = {"type": "console-message", "action": "barrage-diamond-list"}
    if top_n > 0:
        payload["top_n"] = top_n
    async with websockets.connect(URI) as ws:
        await _drain_initial(ws)
        await ws.send(json.dumps(payload, ensure_ascii=False))
        print(f">>> 查询钻石排行 top_n={top_n or 'all'}")

        end = asyncio.get_event_loop().time() + 5.0
        while asyncio.get_event_loop().time() < end:
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=3)
            except asyncio.TimeoutError:
                print("<<< (无响应)")
                return
            try:
                data = json.loads(resp)
            except Exception:
                continue
            if data.get("type") != "barrage-diamond-list":
                continue
            if "error" in data and data.get("data") is None:
                print(f"<<< 错误: {data.get('error')}")
                return
            payload = data["data"]
            print(
                f"<<< 累计用户: {payload['total_users']} 人, "
                f"总钻石: {payload['total_diamonds']}"
            )
            if not payload["ranking"]:
                print("    (本场尚无礼物记录)")
                return
            print(f"    {'排名':<6}{'昵称':<20}{'钻石数':>10}")
            print(f"    {'-' * 6}{'-' * 20}{'-' * 10}")
            for item in payload["ranking"]:
                nick = item["nickname"]
                if len(nick) > 18:
                    nick = nick[:17] + "…"
                print(
                    f"    {item['rank']:<6}{nick:<20}"
                    f"{item['diamonds']:>10}"
                )
            return


def usage():
    print(__doc__)
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        usage()
    op = sys.argv[1]
    if op == "set":
        if len(sys.argv) < 3:
            usage()
        asyncio.run(cmd_set(sys.argv[2:]))
    elif op == "off":
        asyncio.run(cmd_off())
    elif op == "status":
        asyncio.run(cmd_status())
    elif op == "reset-diamond":
        asyncio.run(cmd_reset_diamond())
    elif op == "top":
        top_n = int(sys.argv[2]) if len(sys.argv) >= 3 else 0
        asyncio.run(cmd_top(top_n))
    elif op == "metrics":
        asyncio.run(cmd_metrics())
    elif op == "config-get":
        asyncio.run(cmd_config_get())
    elif op == "config-set":
        if len(sys.argv) < 3:
            print("用法: barrage_custom.py config-set k1=v1 k2=v2 ...")
            sys.exit(1)
        asyncio.run(cmd_config_set(sys.argv[2:]))
    else:
        usage()


if __name__ == "__main__":
    main()

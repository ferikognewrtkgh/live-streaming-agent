"""
test_custom_barrage_queue.py — 自定义弹幕队列离线测试

不依赖 DouyinBarrageGrab 也不依赖前端, 直接构造原始 JSON 消息
喂给 BarrageAdapter._process_raw_message, 验证:
  1. PayLevel / FansClub.Level / DiamondCount 字段抽取
  2. 钻石累计 (Type=5)
  3. 自定义模式开/关切换
  4. 单变量过滤+排序
  5. 多变量级联过滤+排序
  6. 关键词路由
  7. TTL 自动丢弃
  8. 队列上限裁剪
  9. 全关时回退到原有队列
 10. update_custom_config 热更新清空旧队列

运行:
  uv run python test_custom_barrage_queue.py
"""

import asyncio
import json
import sys
from pathlib import Path

# 让 src 可以被 import
sys.path.insert(0, str(Path(__file__).parent))

from src.open_llm_vtuber.barrage_adapter import (
    BarrageAdapter,
    BarrageConfig,
)


# ============================================================
# 工具: 构造原始 DouyinBarrageGrab JSON
# ============================================================

def make_barrage_raw(
    nickname: str,
    content: str,
    pay_level: int = 0,
    fans_level: int = 0,
    sec_uid: str = "",
):
    """模拟 DouyinBarrageGrab 推送的 Type=1 弹幕消息"""
    return json.dumps({
        "Type": 1,
        "Data": {
            "User": {
                "SecUid": sec_uid or f"sec_{nickname}",
                "DisplayId": f"id_{nickname}",
                "Nickname": nickname,
                "PayLevel": pay_level,
                "FansClub": {
                    "ClubName": "test_club",
                    "Level": fans_level,
                },
            },
            "Content": content,
        },
    })


def make_gift_raw(
    nickname: str,
    gift_name: str,
    diamonds: int,
    sec_uid: str = "",
):
    """模拟 DouyinBarrageGrab 推送的 Type=5 礼物消息"""
    return json.dumps({
        "Type": 5,
        "Data": {
            "User": {
                "SecUid": sec_uid or f"sec_{nickname}",
                "DisplayId": f"id_{nickname}",
                "Nickname": nickname,
            },
            "GiftName": gift_name,
            "GiftCount": 1,
            "DiamondCount": diamonds,
        },
    })


# ============================================================
# 测试主体
# ============================================================

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def check(name: str, cond: bool, detail: str = ""):
    tag = PASS if cond else FAIL
    print(f"{tag} {name}  {detail}")
    results.append((name, cond))


async def reset_adapter() -> BarrageAdapter:
    cfg = BarrageConfig(
        # 关掉过滤器里太严的规则,让所有内容都能进入 _enqueue
        min_content_length=1,
        max_content_length=500,
        dedup_window_seconds=0.0,
        max_per_user_in_window=999,
        ignore_exact=[],
        semantic_dedup_window=0.0,
        gift_trigger_min_diamonds=0,
        custom_item_ttl=2.0,
        custom_queue_max_size=10,
    )
    adapter = BarrageAdapter(cfg)
    return adapter


# ------------------------------------------------------------
# Test 1: PayLevel / FansClub.Level / diamond 字段抽取
# ------------------------------------------------------------

async def test_field_extraction():
    print("\n=== Test 1: 字段抽取 ===")
    adapter = await reset_adapter()
    # 启用 wealth 才会走自定义路径,但 BarrageMessage 字段抽取在 _process_raw_message
    # 中无条件执行, 我们用 wealth=0 阈值来观察入队后的字段
    adapter.update_custom_config([
        {"name": "wealth", "enabled": True, "threshold": 0, "priority": 0},
    ])
    await adapter._process_raw_message(
        make_barrage_raw("Alice", "hello", pay_level=18, fans_level=12)
    )
    q = adapter._custom_normal_queue
    check("incoming 弹幕入队", len(q) == 1, f"len={len(q)}")
    if q:
        m = q[0]
        check("PayLevel 抽取 = 18", m.wealth_level == 18,
              f"got {m.wealth_level}")
        check("FansClub.Level 抽取 = 12", m.fan_badge_level == 12,
              f"got {m.fan_badge_level}")


# ------------------------------------------------------------
# Test 2: 钻石累计
# ------------------------------------------------------------

async def test_diamond_accumulation():
    print("\n=== Test 2: 钻石累计 ===")
    adapter = await reset_adapter()
    await adapter._process_raw_message(
        make_gift_raw("Bob", "rose", 100)
    )
    await adapter._process_raw_message(
        make_gift_raw("Bob", "rose2", 250)
    )
    await adapter._process_raw_message(
        make_gift_raw("Carol", "fan", 50)
    )
    totals = adapter._session_diamond_totals
    check("Bob 累计 = 350",
          totals.get("sec_Bob") == 350, f"got {totals.get('sec_Bob')}")
    check("Carol 累计 = 50",
          totals.get("sec_Carol") == 50, f"got {totals.get('sec_Carol')}")


# ------------------------------------------------------------
# Test 3: 自定义模式开关
# ------------------------------------------------------------

async def test_mode_toggle():
    print("\n=== Test 3: 自定义模式开关 ===")
    adapter = await reset_adapter()
    check("默认不激活", adapter._custom_filter_active() is False)

    adapter.update_custom_config([
        {"name": "wealth", "enabled": True, "threshold": 5, "priority": 0},
    ])
    check("启用 wealth 后激活", adapter._custom_filter_active() is True)

    adapter.update_custom_config([
        {"name": "wealth", "enabled": False},
    ])
    check("再关闭 wealth 后回退",
          adapter._custom_filter_active() is False)


# ------------------------------------------------------------
# Test 4: 单变量过滤+排序 (wealth)
# ------------------------------------------------------------

async def test_single_var_filter_sort():
    print("\n=== Test 4: 单变量 wealth 过滤+排序 ===")
    adapter = await reset_adapter()
    adapter.update_custom_config([
        {"name": "wealth", "enabled": True, "threshold": 10, "priority": 0},
    ])
    # wealth=5 应该被过滤
    await adapter._process_raw_message(
        make_barrage_raw("Low", "msg1", pay_level=5)
    )
    # wealth=15, wealth=20, wealth=12 都过, 应该按降序排
    await adapter._process_raw_message(
        make_barrage_raw("Mid", "msg2", pay_level=15)
    )
    await adapter._process_raw_message(
        make_barrage_raw("High", "msg3", pay_level=20)
    )
    await adapter._process_raw_message(
        make_barrage_raw("LowPass", "msg4", pay_level=12)
    )
    q = adapter._custom_normal_queue
    nicks = [m.nickname for m in q]
    check("低 wealth 被过滤",
          adapter._custom_filtered_count >= 1,
          f"filtered={adapter._custom_filtered_count}")
    check("剩余3条", len(q) == 3, f"got {nicks}")
    check("按 wealth 降序排: High>Mid>LowPass",
          nicks == ["High", "Mid", "LowPass"], f"got {nicks}")


# ------------------------------------------------------------
# Test 5: 多变量级联过滤+排序
# ------------------------------------------------------------

async def test_multi_var_filter_sort():
    print("\n=== Test 5: 多变量级联 wealth(p=0) + fan_badge(p=1) ===")
    adapter = await reset_adapter()
    adapter.update_custom_config([
        {"name": "wealth", "enabled": True, "threshold": 10, "priority": 0},
        {"name": "fan_badge", "enabled": True, "threshold": 5, "priority": 1},
    ])
    # A: 通过, wealth=15 badge=6
    await adapter._process_raw_message(
        make_barrage_raw("A", "a", pay_level=15, fans_level=6)
    )
    # B: 通过, wealth=25 badge=6 → 排在 A 之前 (wealth 优先)
    await adapter._process_raw_message(
        make_barrage_raw("B", "b", pay_level=25, fans_level=6)
    )
    # C: 通过, wealth=15 badge=10 → 同 wealth, 比 A badge 高 → 在 A 之前
    await adapter._process_raw_message(
        make_barrage_raw("C", "c", pay_level=15, fans_level=10)
    )
    # D: 被过滤, wealth=8 (虽然 badge=20)
    await adapter._process_raw_message(
        make_barrage_raw("D", "d", pay_level=8, fans_level=20)
    )
    # E: 被过滤, wealth=30 但 badge=3
    await adapter._process_raw_message(
        make_barrage_raw("E", "e", pay_level=30, fans_level=3)
    )

    nicks = [m.nickname for m in adapter._custom_normal_queue]
    check("过滤后只剩 A B C",
          set(nicks) == {"A", "B", "C"}, f"got {nicks}")
    check("排序: B > C > A",
          nicks == ["B", "C", "A"], f"got {nicks}")


# ------------------------------------------------------------
# Test 6: 关键词路由
# ------------------------------------------------------------

async def test_keyword_routing():
    print("\n=== Test 6: 关键词路由 ===")
    adapter = await reset_adapter()
    adapter.update_custom_config([
        {"name": "wealth", "enabled": True, "threshold": 5, "priority": 0},
    ])
    # 默认 keyword_list = ["主播", "zhubo"]
    await adapter._process_raw_message(
        make_barrage_raw("X", "你好啊", pay_level=10)
    )
    await adapter._process_raw_message(
        make_barrage_raw("Y", "主播好棒", pay_level=10)
    )
    check("关键词队列只有 Y",
          [m.nickname for m in adapter._custom_keyword_queue] == ["Y"])
    check("普通队列只有 X",
          [m.nickname for m in adapter._custom_normal_queue] == ["X"])


# ------------------------------------------------------------
# Test 7: TTL 自动丢弃
# ------------------------------------------------------------

async def test_ttl_drop():
    print("\n=== Test 7: TTL 自动丢弃 (custom_item_ttl=2s) ===")
    adapter = await reset_adapter()
    adapter.update_custom_config([
        {"name": "wealth", "enabled": True, "threshold": 0, "priority": 0},
    ])
    await adapter._process_raw_message(
        make_barrage_raw("T1", "msg1", pay_level=10)
    )
    await adapter._process_raw_message(
        make_barrage_raw("T2", "主播", pay_level=10)
    )
    check("入队",
          len(adapter._custom_normal_queue) == 1
          and len(adapter._custom_keyword_queue) == 1)
    print("    sleeping 3s for TTL...")
    await asyncio.sleep(3.0)
    dropped = adapter._sweep_custom_queues_stale()
    check("TTL 清扫 dropped=2", dropped == 2, f"got {dropped}")
    check("两队列均清空",
          len(adapter._custom_normal_queue) == 0
          and len(adapter._custom_keyword_queue) == 0)


# ------------------------------------------------------------
# Test 8: 队列上限裁剪
# ------------------------------------------------------------

async def test_queue_cap():
    print("\n=== Test 8: 队列上限 (custom_queue_max_size=10) ===")
    adapter = await reset_adapter()
    adapter.update_custom_config([
        {"name": "wealth", "enabled": True, "threshold": 0, "priority": 0},
    ])
    # 喂 15 条, wealth 从 1 到 15, 应该只保留最高 10 条 (wealth 6..15)
    for i in range(1, 16):
        await adapter._process_raw_message(
            make_barrage_raw(f"U{i}", f"m{i}", pay_level=i)
        )
    q = adapter._custom_normal_queue
    check("队列长度限制在 10", len(q) == 10, f"got {len(q)}")
    wealth_levels = [m.wealth_level for m in q]
    check("保留的是最高 10 条 (6..15)",
          sorted(wealth_levels) == list(range(6, 16)),
          f"got {sorted(wealth_levels)}")
    check("最前面是 wealth=15",
          q[0].wealth_level == 15, f"got {q[0].wealth_level}")


# ------------------------------------------------------------
# Test 9: 全关时回退到原有队列
# ------------------------------------------------------------

async def test_fallback_to_original():
    print("\n=== Test 9: 全关时回退原有队列 ===")
    adapter = await reset_adapter()
    # 默认全关
    await adapter._process_raw_message(
        make_barrage_raw("Z", "hello", pay_level=5, fans_level=2)
    )
    check("自定义队列空",
          len(adapter._custom_normal_queue) == 0
          and len(adapter._custom_keyword_queue) == 0)
    check("原有普通队列有 1 条",
          adapter.barrage_queue_normal.qsize() == 1,
          f"got {adapter.barrage_queue_normal.qsize()}")


# ------------------------------------------------------------
# Test 10: 热更新清空旧队列
# ------------------------------------------------------------

async def test_hot_reload_clear():
    print("\n=== Test 10: 热更新时清空旧队列 ===")
    adapter = await reset_adapter()
    adapter.update_custom_config([
        {"name": "wealth", "enabled": True, "threshold": 5, "priority": 0},
    ])
    await adapter._process_raw_message(
        make_barrage_raw("Q", "msg", pay_level=10)
    )
    check("更新前队列有内容", len(adapter._custom_normal_queue) == 1)
    # 改阈值, 旧队列应被清空
    adapter.update_custom_config([
        {"name": "wealth", "enabled": True, "threshold": 50, "priority": 0},
    ])
    check("热更新后旧队列被清空",
          len(adapter._custom_normal_queue) == 0)


# ------------------------------------------------------------
# Test 11: 钻石排行筛选 (diamond_rank)
# ------------------------------------------------------------

async def test_diamond_rank_filter():
    print("\n=== Test 11: diamond_rank 筛选 ===")
    adapter = await reset_adapter()
    # 先制造礼物累计
    await adapter._process_raw_message(make_gift_raw("Rich", "g", 1000))
    await adapter._process_raw_message(make_gift_raw("Poor", "g", 50))
    # 启用 diamond_rank 阈值 500
    adapter.update_custom_config([
        {"name": "diamond_rank", "enabled": True,
         "threshold": 500, "priority": 0},
    ])
    await adapter._process_raw_message(
        make_barrage_raw("Rich", "hi", sec_uid="sec_Rich")
    )
    await adapter._process_raw_message(
        make_barrage_raw("Poor", "hi", sec_uid="sec_Poor")
    )
    nicks = [m.nickname for m in adapter._custom_normal_queue]
    check("只有 Rich 通过", nicks == ["Rich"], f"got {nicks}")


# ------------------------------------------------------------
# Test 12: B2 - LRU 钻石累计
# ------------------------------------------------------------

async def test_diamond_lru():
    print("\n=== Test 12: B2 钻石累计 LRU ===")
    from src.open_llm_vtuber.barrage_adapter import (
        BarrageAdapter, BarrageConfig,
    )
    adapter = BarrageAdapter(BarrageConfig(
        gift_trigger_min_diamonds=0,
        diamond_max_users=3,  # 极小上限便于测试
    ))
    for nick in ["A", "B", "C", "D", "E"]:
        await adapter._process_raw_message(
            make_gift_raw(nick, "g", 100, sec_uid=f"sec_{nick}")
        )
    totals = adapter._session_diamond_totals
    check("LRU 上限生效, 只剩 3 个", len(totals) == 3,
          f"got {len(totals)}")
    check("最老的 A B 被淘汰",
          "sec_A" not in totals and "sec_B" not in totals,
          f"got keys {list(totals.keys())}")
    check("最新的 C D E 保留",
          all(f"sec_{n}" in totals for n in ["C", "D", "E"]),
          f"got keys {list(totals.keys())}")
    check("淘汰计数 = 2",
          adapter._diamond_evicted_count == 2,
          f"got {adapter._diamond_evicted_count}")


# ------------------------------------------------------------
# Test 13: B1 - 礼物连击合并
# ------------------------------------------------------------

async def test_gift_combo_merge():
    print("\n=== Test 13: B1 礼物连击合并 ===")
    from src.open_llm_vtuber.barrage_adapter import (
        BarrageAdapter, BarrageConfig,
    )
    adapter = BarrageAdapter(BarrageConfig(
        gift_trigger_min_diamonds=0,
        gift_dedup_window=0.0,
        gift_combo_window=1.0,
        gift_combo_max_wait=10.0,
    ))
    # 同用户同礼物 5 连击
    for _ in range(5):
        await adapter._process_raw_message(
            make_gift_raw("BigBro", "嘉年华", 1000, sec_uid="sec_BigBro")
        )
    # 此时应该在 combo buffer 里, 还没 flush
    check("缓冲区 1 条",
          len(adapter._gift_combo_buffer) == 1,
          f"got {len(adapter._gift_combo_buffer)}")
    entry = list(adapter._gift_combo_buffer.values())[0]
    check("累加 count=5",
          entry["msg"].raw_data["_accumulated_count"] == 5,
          f"got {entry['msg'].raw_data['_accumulated_count']}")
    check("累加 diamonds=5000",
          entry["msg"].raw_data["_accumulated_diamonds"] == 5000,
          f"got {entry['msg'].raw_data['_accumulated_diamonds']}")
    # 等过 combo window 后 flush
    print("    等 1.5s 触发 flush...")
    await asyncio.sleep(1.5)
    await adapter._flush_gift_combos()
    check("缓冲清空", len(adapter._gift_combo_buffer) == 0)
    check("礼物入队 1 条 (合并后)",
          adapter.gift_queue.qsize() == 1,
          f"got {adapter.gift_queue.qsize()}")
    flushed = await adapter.gift_queue.get()
    check("入队消息含连击数 5x", "5x" in flushed.content,
          f"got: {flushed.content}")
    check("入队消息含累计钻石 5000",
          "5000" in flushed.content,
          f"got: {flushed.content}")


# ------------------------------------------------------------
# Test 14: B3 - 进直播间欢迎语聚合
# ------------------------------------------------------------

async def test_welcome_batch():
    print("\n=== Test 14: B3 进直播间欢迎语聚合 ===")
    from src.open_llm_vtuber.barrage_adapter import (
        BarrageAdapter, BarrageConfig,
    )
    adapter = BarrageAdapter(BarrageConfig(
        welcome_enabled=True,
        welcome_batch_window=0.5,
        welcome_max_batch=3,
        min_content_length=1,
        ignore_exact=[],
    ))
    # 模拟 Type=4 进直播间消息
    def join_raw(nick):
        return json.dumps({
            "Type": 4,
            "Data": {
                "User": {
                    "SecUid": f"sec_{nick}",
                    "DisplayId": f"id_{nick}",
                    "Nickname": nick,
                },
            },
        })

    for nick in ["甲", "乙", "丙", "丁", "戊"]:
        await adapter._process_raw_message(join_raw(nick))

    check("待欢迎缓冲 5 人", len(adapter._pending_joins) == 5,
          f"got {len(adapter._pending_joins)}")
    # 触发立即 flush (满 max_batch=3)
    await adapter._flush_event_batches()
    check("一批 3 人后, 剩 2 人",
          len(adapter._pending_joins) == 2,
          f"got {len(adapter._pending_joins)}")
    # 检查入队的欢迎消息
    items = []
    while not adapter.barrage_queue_normal.empty():
        items.append(adapter.barrage_queue_normal.get_nowait())
    check("普通队列有 1 条欢迎消息", len(items) == 1,
          f"got {len(items)}")
    if items:
        check("欢迎消息包含 3 个昵称",
              "甲" in items[0].content
              and "乙" in items[0].content
              and "丙" in items[0].content,
              f"got: {items[0].content}")


# ------------------------------------------------------------
# Test 15: B3 - Type=3 点赞计数 + Type=8 心跳
# ------------------------------------------------------------

async def test_likes_and_heartbeat():
    print("\n=== Test 15: B3 点赞累计 + Type=8 心跳 ===")
    adapter = await reset_adapter()
    # 点赞: Type=3, 字段 Count
    def like_raw(count):
        return json.dumps({
            "Type": 3,
            "Data": {
                "User": {"Nickname": "x"},
                "Count": count,
            },
        })

    await adapter._process_raw_message(like_raw(10))
    await adapter._process_raw_message(like_raw(5))
    check("点赞计数 = 15", adapter._like_count == 15,
          f"got {adapter._like_count}")

    # Type=8 直播间统计
    stats_raw = json.dumps({
        "Type": 8,
        "Data": {"MemberCount": 1234, "TotalUserCount": 56789},
    })
    await adapter._process_raw_message(stats_raw)
    check("Type=8 触发心跳时间戳",
          adapter._last_heartbeat_time > 0)
    check("Type=8 记录 member_count=1234",
          adapter._last_room_stats.get("member_count") == 1234,
          f"got {adapter._last_room_stats}")


# ------------------------------------------------------------
# Test 16: B4 - 运行时配置热更新
# ------------------------------------------------------------

async def test_runtime_config():
    print("\n=== Test 16: B4 运行时配置热更新 ===")
    adapter = await reset_adapter()
    # 改简单字段
    result = adapter.update_runtime_config({
        "consume_interval": 7.5,
        "welcome_enabled": True,
        "keyword_list": ["主播", "宝宝", "姐姐"],
        "evil_unknown_field": "should_be_skipped",
    })
    check("consume_interval 已生效",
          adapter.config.consume_interval == 7.5)
    check("welcome_enabled 已生效",
          adapter.config.welcome_enabled is True)
    check("keyword_list 已生效",
          adapter.config.keyword_list == ["主播", "宝宝", "姐姐"])
    check("_keywords_lower 同步重建",
          adapter._keywords_lower == ["主播", "宝宝", "姐姐"])
    check("未知字段被跳过",
          "evil_unknown_field" in result["skipped"],
          f"skipped: {result['skipped']}")
    # 改 filter 相关 → 应触发 filter 重建
    old_filter_id = id(adapter.filter)
    adapter.update_runtime_config({"min_content_length": 5})
    check("min_content_length 已生效",
          adapter.config.min_content_length == 5)
    check("BarrageFilter 已重建",
          id(adapter.filter) != old_filter_id)


# ------------------------------------------------------------
# Test 17: B9 - metrics 速率
# ------------------------------------------------------------

async def test_metrics():
    print("\n=== Test 17: B9 metrics ===")
    adapter = await reset_adapter()
    adapter.update_custom_config([
        {"name": "wealth", "enabled": True, "threshold": 5, "priority": 0},
    ])
    # 喂 4 条弹幕, 1 条会被过滤; 用不同内容避免语义去重
    await adapter._process_raw_message(
        make_barrage_raw("A", "今天天气真好啊", pay_level=10)
    )
    await adapter._process_raw_message(
        make_barrage_raw("B", "主播在干嘛", pay_level=10)
    )
    await adapter._process_raw_message(
        make_barrage_raw("C", "晚上吃了什么", pay_level=10)
    )
    await adapter._process_raw_message(
        make_barrage_raw("LowWealth", "我是路过的", pay_level=1)
    )

    m = adapter.get_metrics()
    check("recv 计 4", len(adapter._metrics_recv) == 4,
          f"got {len(adapter._metrics_recv)}")
    check("enqueue 计 3", len(adapter._metrics_enqueue) == 3,
          f"got {len(adapter._metrics_enqueue)}")
    check("filter_drop 计 1", len(adapter._metrics_filter_drop) == 1,
          f"got {len(adapter._metrics_filter_drop)}")
    check("queues 字段存在",
          "barrage_high" in m["queues"]
          and "custom_normal" in m["queues"])
    check("counters 字段存在",
          "total_received" in m["counters"]
          and "diamond_evicted_total" in m["counters"])


# ============================================================
# 入口
# ============================================================

async def main():
    print("=" * 60)
    print(" 自定义弹幕队列离线测试")
    print("=" * 60)

    tests = [
        test_field_extraction,
        test_diamond_accumulation,
        test_mode_toggle,
        test_single_var_filter_sort,
        test_multi_var_filter_sort,
        test_keyword_routing,
        test_ttl_drop,
        test_queue_cap,
        test_fallback_to_original,
        test_hot_reload_clear,
        test_diamond_rank_filter,
        test_diamond_lru,
        test_gift_combo_merge,
        test_welcome_batch,
        test_likes_and_heartbeat,
        test_runtime_config,
        test_metrics,
    ]
    for t in tests:
        try:
            await t()
        except Exception as e:
            check(f"{t.__name__} 抛异常", False, str(e))
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f" 结果: {passed}/{total} 通过")
    if passed != total:
        print(" 失败项:")
        for name, ok in results:
            if not ok:
                print(f"   - {name}")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

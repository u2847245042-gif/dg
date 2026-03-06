#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
import json
import os
import signal
import subprocess
import sys

import websockets

# 已连接客户端集合
connected_clients = set()
# 退出事件（收到 cmd=exit 时触发），在 main() 里初始化
shutdown_event = None
# 保存子进程句柄
dg_process = None
name = "dg_gtt.py"
# ======================== 子进程启动/停止 ==========================
def start_dg():
    global dg_process
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, name)

    if not os.path.exists(script_path):
        print(f"[ws_server] 找不到 {name}：{script_path}")
        return None

    # 如果已经在跑，就不重复启动
    if dg_process is not None and dg_process.poll() is None:
        print(f"[ws_server] {name}已在运行中，跳过启动")
        return dg_process

    python_exec = sys.executable  # 当前 Python 解释器
    print(f"[ws_server] 启动子进程: {python_exec} {script_path}")

    dg_process = subprocess.Popen(
        [python_exec, script_path],
        cwd=script_dir,
        start_new_session=True,
    )
    print(f"[ws_server] {name}已启动，PID = {dg_process.pid}")
    return dg_process
async def stop_dg():
    global dg_process
    if dg_process is None:
        return

    if dg_process.poll() is None:  # 仍在运行
        try:
            os.killpg(os.getpgid(dg_process.pid), signal.SIGTERM)
        except Exception:
            try:
                dg_process.terminate()
            except Exception:
                pass
        await asyncio.sleep(0.8)
        if dg_process.poll() is None:
            try:
                os.killpg(os.getpgid(dg_process.pid), signal.SIGKILL)
            except Exception:
                try:
                    dg_process.kill()
                except Exception:
                    pass

    print(f"[ws_server] {name}已结束")
    dg_process = None
# ======================== WebSocket 处理 ===========================
async def handler(websocket):
    """
    每个客户端连接的处理协程。
    """
    global shutdown_event, dg_process

    print("[ws_server] client connected:", websocket.remote_address)
    connected_clients.add(websocket)

    # ⭐ 第一个客户端上线时启动
    if dg_process is None or dg_process.poll() is not None:
        print(f"[ws_server] 第一个客户端上线，启动 {name}...")
        start_dg()

    try:
        async for msg in websocket:
            # print("[ws_server] recv:", msg)

            # 尝试解析 JSON，看是否为退出指令
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                data = None
            # ↓↓↓ 新增：识别 {"marked": False} 并广播清零指令
            if isinstance(data, dict) and data.get("marked") is False:
                await asyncio.gather(
                    *[ws.send(json.dumps({"cmd": "clear_marked"})) for ws in connected_clients],
                    return_exceptions=True
                )
                print("[SERVER] 已广播 clear_marked")
                continue   # 不再向下转发
            # ↑↑↑ 插入结束
            if isinstance(data, dict):
                cmd = data.get("cmd")
                if cmd == "q":
                    print(json.dumps({"event": "back_to_menu"}), flush=True)
                    print("[ws_server] 收到 q，通知主进程返回菜单")
                    continue
                if cmd == "exit":
                    # 收到退出指令
                    print(f"[ws_server] 收到退出指令 cmd=exit，准备关闭服务器和 {name}")
                    if shutdown_event is not None:
                        shutdown_event.set()
                    break

            # 普通消息：广播给其他客户端
            dead = []
            for ws in connected_clients:
                if ws is websocket:
                    continue
                try:
                    await ws.send(msg)
                except Exception:
                    dead.append(ws)

            # 清理断开的连接
            for ws in dead:
                connected_clients.discard(ws)

    except websockets.ConnectionClosed:
        print("[ws_server] client disconnected:", websocket.remote_address)
    finally:
        connected_clients.discard(websocket)
async def main():
    global shutdown_event

    # 在当前 event loop 上创建 Event，避免 “不同 loop” 报错
    shutdown_event = asyncio.Event()

    # 启动 stdin 读取任务
    asyncio.create_task(stdin_reader())

    # 注意：这里不再一启动就 start_dg，
    # 而是等第一个客户端连上时在 handler 里启动。

    # 启动 WebSocket 服务器
    ip_id = 8769
    async with websockets.serve(handler, "0.0.0.0", ip_id, reuse_address=True,origins=None):
        print(f"[ws_server] WebSocket server started at ws://0.0.0.0:{ip_id}")
        print("[ws_server] 等待客户端连接中...")

        # 等待退出事件（比如收到 {"cmd":"exit"}）
        await shutdown_event.wait()

    print("[ws_server] WebSocket server 即将关闭...")
    # 关闭所有还活着的连接
    for ws in list(connected_clients):
        try:
            await ws.close()
        except Exception:
            pass
    connected_clients.clear()

    # 停掉子进程
    await stop_dg()
    print("[ws_server] 退出完成，再见。")

async def stdin_reader():
    """读取标准输入指令"""
    global shutdown_event
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    
    while True:
        line = await reader.readline()
        if not line:
            break
        try:
            data = json.loads(line.decode().strip())
            if data.get("cmd") in ["exit", "shutdown"]:
                print(f"[ws_server] 收到 stdin 指令: {data.get('cmd')}，准备退出")
                if shutdown_event:
                    shutdown_event.set()
                break
        except:
            pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[ws_server] 收到 Ctrl+C，中断，正在清理...")
        try:
            asyncio.run(stop_dg())
        except Exception:
            pass
        print("[ws_server] 已退出")

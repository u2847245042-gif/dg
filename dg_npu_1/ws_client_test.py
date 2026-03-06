import asyncio
import websockets
import sys


async def run_client():
    uri = "ws://localhost:8765"
    try:
        async with websockets.connect(uri) as websocket:
            print(f"已连接到服务端: {uri}")
            print("可用指令:")
            print("  1    - 暂时关闭摄像头并运行外部脚本")
            print("  q    - 结束外部脚本并打开摄像头")
            print("  exit - 结束所有连接并关闭服务端")
            print("  ctrl+c - 退出客户端")

            while True:
                # 使用 run_in_executor 来避免 input 阻塞事件循环
                loop = asyncio.get_event_loop()
                cmd = await loop.run_in_executor(None, sys.stdin.readline)
                cmd = cmd.strip()

                if not cmd:
                    continue

                await websocket.send(cmd)
                response = await websocket.recv()
                print(f"服务端响应: {response}")

                if cmd.lower() == "exit":
                    print("客户端退出")
                    break

    except ConnectionRefusedError:
        print("错误: 无法连接到服务端，请确保 z-server.py 已启动。")
    except websockets.exceptions.ConnectionClosed:
        print("服务端连接已关闭")
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(run_client())
    except KeyboardInterrupt:
        print("\n客户端已停止")

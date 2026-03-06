import asyncio
import websockets
import json
import time

async def test():
    uri = "ws://127.0.0.1:8768"
    async with websockets.connect(uri) as websocket:
        print("connected to server")

        # 持续接收实时状态
        async for msg in websocket:
            data = json.loads(msg)
            print("recv:", data)
            # if data["count"] == 20:
            #     # 发送控制指令
            #     await websocket.send(json.dumps({"marked": False}))
            #     print("sent: {'marked': False}")

            # await asyncio.sleep(60)
            # # 发送控制指令
            # await websocket.send(json.dumps({"marked": False}))
            # print("sent: {'marked': False}")



asyncio.run(test())
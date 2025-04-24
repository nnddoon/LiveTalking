import requests
import json
import sys
import aiohttp
import asyncio

server_url = "http://117.50.245.121:8010"
session_id = 990660# 替換為你的有效會話 ID

# 準備數據
# data = {
#     "sessionid": session_id,
#     "type": "echo",
#     "interrupt": True,
#     "text": "带娃家庭看过来！68 元套餐有肉有菜有饮料，孩子爱吃还省钱！点两份外卖要 100 多，这里直接省出一张电影票！约会不知道吃啥？直接囤券！扫码即用，不用纠结点单，把钱花在刀刃上！家人们看这个咕嘟咕嘟冒泡的牛腩煲！老茶头牛腩煲慢炖 好几个小时，每块牛腩都吸饱了琥珀色茶汤！（夹起带筋牛腩）瞧瞧这 Q 弹到会跳舞的筋膜！轻轻一抿就化开的软烂肉质！这浓到挂勺的茶油汤底，泡饭能炫三碗！吸满汤汁的萝卜比肉还香！今天买团购券全国 门店通用，随时去随时核销！节假日通用！我们特意跟商家申请了周末不加价政策！快囤"
# }

data = {
    "sessionid": session_id,
    "type": "echo",
    "interrupt": True,
    "text": ""
}

# 發送請求
try:
    headers = {"Content-Type": "application/json"}
    response = requests.post(f"{server_url}/human", 
                           json=data,  # 使用 json 參數而不是 data
                           headers=headers)
    
    print(f"狀態碼: {response.status_code}")
    print(f"響應內容: {response.text}")
    
except Exception as e:
    print(f"發生錯誤: {e}") 



# async def upload_audio(server_url, session_id, audio_file_path):
#     """上傳音頻檔案到服務器"""
#     data = aiohttp.FormData()
#     data.add_field('sessionid', str(session_id))
#     data.add_field('file', 
#                    open(audio_file_path, 'rb'),
#                    filename=audio_file_path.split('/')[-1],
#                    content_type='audio/wav')  # 根據實際音頻格式調整

#     async with aiohttp.ClientSession() as session:
#         async with session.post(f"{server_url}/humanaudio", data=data) as response:
#             if response.status == 200:
#                 result = await response.json()
#                 if result["code"] == 0:
#                     print("音頻上傳成功")
#                 else:
#                     print(f"上傳失敗: {result['msg']}")
#             else:
#                 print(f"請求失敗，狀態碼: {response.status}")

# async def main():
#     server_url = "http://localhost:8010"  # 替換為你的服務器 URL
#     audio_file = r"E:\Ultralight-Digital-Human\1741347995294078570-244920627826779.mp3"  # 替換為你的音頻檔案路徑
    
#     # 1. 建立連接並獲取 session ID
#     # session_id = await create_webrtc_connection(server_url)
#     # if not session_id:
#     #     print("無法獲取 session ID，退出")
#     #     return
    
#     # print(f"成功獲取 session ID: {session_id}")
    
#     # 2. 上傳音頻檔案
#     await upload_audio(server_url, session_id, audio_file)
    
#     # 3. 檢查是否在說話
#     async with aiohttp.ClientSession() as session:
#         async with session.post(f"{server_url}/is_speaking", 
#                               json={"sessionid": session_id}) as response:
#             result = await response.json()
#             print(f"數字人說話狀態: {result['data']}")

# if __name__ == "__main__":
#     asyncio.run(main())
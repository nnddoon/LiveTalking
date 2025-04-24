import requests
import os
import subprocess

def optimize_audio_for_upload(input_file, max_size_mb=1):
    """優化音頻文件大小以適合上傳"""
    # 檢查文件是否存在
    if not os.path.exists(input_file):
        print(f"文件不存在: {input_file}")
        return None
        
    # 檢查文件大小
    file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
    print(f"原始文件大小: {file_size_mb:.2f} MB")
    
    # 如果文件已經夠小，直接返回原文件
    if file_size_mb <= max_size_mb:
        print("文件大小已經在限制範圍內，無需壓縮")
        return input_file
        
    # 創建輸出文件名
    file_name, file_ext = os.path.splitext(input_file)
    output_file = f"{file_name}_compressed.mp3"
    
    # 使用 ffmpeg 壓縮文件 (轉換為較低比特率的 MP3)
    ffmpeg_cmd = ["ffmpeg", "-i", input_file, "-b:a", "32k", "-ac", "1", output_file]
    subprocess.run(ffmpeg_cmd)
    
    # 檢查輸出文件大小
    output_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"壓縮後文件大小: {output_size_mb:.2f} MB")
    
    return output_file

def upload_audio(server_url, session_id, audio_file):
    """上傳音頻文件到服務器"""
    # 確保文件夠小
    optimized_file = optimize_audio_for_upload(audio_file, max_size_mb=1)
    if not optimized_file:
        return False
        
    # 上傳優化後的文件
    files = {'file': open(optimized_file, 'rb')}
    data = {'sessionid': session_id, 'interrupt':False}
    
    response = requests.post(f"{server_url}/humanaudio", files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        if result["code"] == 0:
            print("音頻上傳成功")
            return True
        else:
            print(f"上傳失敗: {result.get('msg', '未知錯誤')}")
    else:
        # print(f"請求失敗，狀態碼: {response.status_code}")
        print(f"請求失敗，狀態碼: {response.status_code}, 響應內容: {response.text}")
        
    return False

# 使用示例
# server_url = "http://localhost:8000"
server_url = "http://localhost:8010" 
session_id = 903885
file_path = r"D:\Downloads\1743257640690013294-252742721798432.mp3"

upload_audio(server_url, session_id, file_path) 
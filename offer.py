import asyncio
import json
import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCIceCandidate
from aiortc.contrib.media import MediaPlayer, MediaRecorder

# Track for receiving video
class VideoReceiver(MediaStreamTrack):
    kind = "video"
    
    def __init__(self):
        super().__init__()
        self.frames = []
    
    async def recv(self):
        # In a real application, you would process video frames here
        if not self.frames:
            frame = await asyncio.sleep(0.1)
        else:
            frame = self.frames.pop(0)
        return frame

# Track for receiving audio
class AudioReceiver(MediaStreamTrack):
    kind = "audio"
    
    def __init__(self):
        super().__init__()
        self.frames = []
    
    async def recv(self):
        # In a real application, you would process audio frames here
        if not self.frames:
            frame = await asyncio.sleep(0.1)
        else:
            frame = self.frames.pop(0)
        return frame

async def create_webrtc_connection(server_url="http://localhost:8080"):
    # Create a new RTCPeerConnection
    pc = RTCPeerConnection()
    
    # Add transceivers for both audio and video
    # This tells the server we want to receive both audio and video
    pc.addTransceiver("audio", direction="recvonly")
    pc.addTransceiver("video", direction="recvonly")
    
    # Create and set local description (the offer)
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    
    # Print the SDP to see what it looks like
    print("Generated SDP:")
    print(pc.localDescription.sdp)
    
    # The offer.sdp now contains the SDP in the correct format
    offer_data = {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{server_url}/offer",
                json=offer_data
            ) as response:
                if response.status == 200:
                    answer_data = await response.json()
                    
                    # The response should contain:
                    # - type: "answer"
                    # - sdp: the answer SDP
                    # - sessionid: unique session identifier
                    
                    print(f"Received session ID: {answer_data['sessionid']}")
                    
                    # Set the remote description with the answer from server
                    answer = RTCSessionDescription(
                        sdp=answer_data["sdp"],
                        type=answer_data["type"]
                    )
                    await pc.setRemoteDescription(answer)
                    
                    return pc, answer_data["sessionid"]
                else:
                    print(f"Error: Server returned status {response.status}")
                    return None, None
    except Exception as e:
        print(f"Error establishing connection: {e}")
        return None, None

async def interrupt_speech(server_url, sessionid):
    """Send an interrupt command to stop the current speech"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{server_url}/human",
                json={
                    "sessionid": sessionid,
                    "text": "",  # Empty text since we just want to interrupt
                    "type": "chat",
                    "interrupt": True  # This triggers the interrupt
                }
            ) as response:
                result = await response.json()
                if result["code"] == 0:
                    print("Successfully interrupted speech")
                else:
                    print(f"Failed to interrupt speech: {result}")
    except Exception as e:
        print(f"Error sending interrupt command: {e}")

async def main():
    # Server URL - update this to match your server
    server_url = "http://192.168.2.126:8010"  # Use the port from app.py (opt.listenport)
    
    # Create a new RTCPeerConnection
    pc = RTCPeerConnection()
    
    # Add transceivers for both audio and video (to receive media)
    pc.addTransceiver("audio", direction="recvonly")
    pc.addTransceiver("video", direction="recvonly")
    
    # Create and set local description (the offer)
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    
    # Print the SDP to see what it looks like
    print("Generated SDP:")
    print(pc.localDescription.sdp)
    
    # Prepare the offer data to send to the server
    offer_data = {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }
    
    try:
        # Send the offer to the server
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{server_url}/offer",
                json=offer_data
            ) as response:
                if response.status == 200:
                    # Parse the server's answer
                    answer_data = await response.json()
                    
                    # Print the session ID (important for other API calls)
                    print(f"Received session ID: {answer_data['sessionid']}")
                    
                    # Set the remote description with the answer from server
                    answer = RTCSessionDescription(
                        sdp=answer_data["sdp"],
                        type=answer_data["type"]
                    )
                    await pc.setRemoteDescription(answer)
                    
                    # Now the WebRTC connection is established!
                    print("WebRTC connection established!")
                    
                    # Example: Send a chat message using the session ID
                    sessionid = answer_data['sessionid']
                    # await send_chat_message(server_url, sessionid, "Hello, this is a test message")
                    
                    # Wait for a while
                    await asyncio.sleep(5)
                    
                    # Send interrupt command
                    await interrupt_speech(server_url, sessionid)
                    
                    # Keep the connection alive for a while
                    await asyncio.sleep(30)
                    
                    # Close the connection when done
                    await pc.close()
                    print("Connection closed")
                else:
                    print(f"Error: Server returned status {response.status}")
    except Exception as e:
        print(f"Error establishing connection: {e}")

# async def send_chat_message(server_url, sessionid, text):
#     """Send a chat message to the server using the established session"""
#     try:
#         async with aiohttp.ClientSession() as session:
#             async with session.post(
#                 f"{server_url}/human",
#                 json={
#                     "sessionid": sessionid,
#                     "text": text,
#                     "type": "chat",
#                     "interrupt": False
#                 }
#             ) as response:
#                 result = await response.json()
#                 print(f"Chat message response: {result}")
#     except Exception as e:
#         print(f"Error sending chat message: {e}")

if __name__ == "__main__":
    asyncio.run(main())
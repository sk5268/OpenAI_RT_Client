import os
import json
import asyncio
import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.mediastreams import MediaStreamError, AudioStreamTrack
import time
import sys

# Get API key from environment variable instead of hardcoding
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    exit(1)

model = "gpt-4o-mini-realtime-preview"
url = f"https://api.openai.com/v1/realtime?model={model}"
headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/sdp"
}

# Variable to accumulate the complete response
complete_response = ""

# Minimal silent audio track - required for WebRTC connection
class SilentAudioStreamTrack(AudioStreamTrack):
    """
    A silent audio track required by OpenAI's WebRTC implementation.
    This is the minimum required to establish a connection but we won't process audio.
    """
    def __init__(self):
        super().__init__()
        self._timestamp = 0
        self._samples = 1024

    async def recv(self):
        from aiortc.mediastreams import AudioFrame
        import numpy as np
        
        # Create a silent audio frame
        data = np.zeros(self._samples, dtype=np.float32)
        frame = AudioFrame(
            channels=1,
            data=data,
            sample_rate=48000,
            timestamp=self._timestamp
        )
        self._timestamp += self._samples
        return frame

# Handler for messages received on the data channel - focused on text only
def on_message(message):
    global complete_response
    try:
        server_event = json.loads(message)
        event_type = server_event.get("type", "")
        
        if event_type == "response.created":
            complete_response = ""  # Reset the response buffer
        # Handle streaming text deltas - accumulate rather than display
        elif event_type == "response.text.delta":
            delta = server_event.get("delta", "")
            complete_response += delta  # Accumulate the response
        # When response is complete, display full response
        elif event_type == "response.done":
            print(complete_response)
            
    except Exception as e:
        pass

async def run_webrtc_client():
    # Create a new RTCPeerConnection
    pc = RTCPeerConnection()
    
    # Add silent audio track - required by OpenAI API but we won't process audio
    silent_track = SilentAudioStreamTrack()
    pc.addTrack(silent_track)
    
    # Create a data channel for sending/receiving text events
    dc = pc.createDataChannel("events")
    
    @dc.on("open")
    def on_open():
        # Send user message as text
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "What Prince album sold the most copies?",
                    }
                ]
            }
        }
        dc.send(json.dumps(event))
        
        # Request text-only response
        event = {
            "type": "response.create",
            "response": {
                "modalities": ["text"]  # Explicitly only request text
            }
        }
        dc.send(json.dumps(event))
    
    @dc.on("message")
    def on_datachannel_message(message):
        on_message(message)
    
    # Create an offer to send to the server
    await pc.setLocalDescription(await pc.createOffer())
    
    # Send the offer to the server and get the answer
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=pc.localDescription.sdp) as response:
            if response.status not in [200, 201]:  # Accept both 200 OK and 201 Created
                return
            
            # Get the SDP answer from the server
            sdp_answer = await response.text()
            
            # Set the remote description with the server's answer
            await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp_answer, type="answer"))
    
    # Wait for the connection to close
    await asyncio.sleep(30)
    
    # Close the peer connection
    await pc.close()

if __name__ == "__main__":
    try:
        # Run the asyncio event loop
        asyncio.run(run_webrtc_client())
    except KeyboardInterrupt:
        pass
    except Exception:
        pass
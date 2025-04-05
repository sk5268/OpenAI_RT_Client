import os
import json
import asyncio
import base64
import wave
import tempfile
import aiohttp
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRecorder
from aiortc.mediastreams import AudioStreamTrack

# Get API key from environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY environment variable is not set")
    exit(1)

# Class for processing audio files
class AudioFileTrack(AudioStreamTrack):
    """
    An audio track that reads from a wav file and sends the data.
    """
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self._file = None
        self._timestamp = 0
        self._sample_rate = 48000
        self._samples_per_frame = 960  # 20ms at 48kHz
        self._file_loaded = False
        self._audio_data = None
        self._position = 0
        self._load_audio_file()
    
    def _load_audio_file(self):
        """Load the audio file into memory and convert it to the right format."""
        try:
            with wave.open(self.file_path, 'rb') as wav_file:
                # Get file properties
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                original_sample_rate = wav_file.getframerate()
                
                # Read all frames and convert to numpy array
                frames = wav_file.readframes(wav_file.getnframes())
                
                # Convert bytes to numpy array based on sample width
                if sample_width == 2:  # 16-bit audio
                    audio_data = np.frombuffer(frames, dtype=np.int16)
                    # Normalize to float32 between -1.0 and 1.0
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif sample_width == 4:  # 32-bit audio
                    audio_data = np.frombuffer(frames, dtype=np.int32)
                    audio_data = audio_data.astype(np.float32) / 2147483648.0
                else:
                    raise ValueError(f"Unsupported sample width: {sample_width}")
                
                # If stereo, convert to mono by averaging channels
                if channels == 2:
                    audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                
                # Resample if necessary (simple method - for production use librosa or similar)
                if original_sample_rate != self._sample_rate:
                    # For simplicity, we'll use a basic resampling technique
                    # In production, you should use a proper resampling library
                    ratio = self._sample_rate / original_sample_rate
                    new_length = int(len(audio_data) * ratio)
                    indices = np.round(np.linspace(0, len(audio_data) - 1, new_length)).astype(int)
                    audio_data = audio_data[indices]
                
                self._audio_data = audio_data
                self._file_loaded = True
                print(f"Loaded audio file: {self.file_path} (length: {len(self._audio_data)/self._sample_rate:.2f}s)")
        except Exception as e:
            print(f"Error loading audio file: {e}")
            # Create empty audio data as fallback
            self._audio_data = np.zeros(self._sample_rate * 5, dtype=np.float32)  # 5 seconds of silence
            self._file_loaded = True
    
    async def recv(self):
        from aiortc.mediastreams import AudioFrame
        
        # If we have no more audio data, send silence
        if self._position >= len(self._audio_data):
            # Create a silent frame
            data = np.zeros(self._samples_per_frame, dtype=np.float32)
        else:
            # Get the next chunk of audio data
            end_pos = min(self._position + self._samples_per_frame, len(self._audio_data))
            data = self._audio_data[self._position:end_pos]
            
            # If we don't have enough samples, pad with zeros
            if len(data) < self._samples_per_frame:
                padding = np.zeros(self._samples_per_frame - len(data), dtype=np.float32)
                data = np.concatenate([data, padding])
            
            self._position += self._samples_per_frame
        
        # Create and return audio frame
        frame = AudioFrame(
            channels=1,
            data=data,
            sample_rate=self._sample_rate,
            timestamp=self._timestamp
        )
        self._timestamp += self._samples_per_frame
        return frame

# Global flags for tracking audio response
received_audio = False
recording_started = False
recorder = None
dc = None
response_started = False
response_done = False

async def run_webrtc_client(input_file, output_file, model="gpt-4o-realtime-preview", voice="alloy"):
    """
    Sends a voice file to OpenAI's Realtime API and saves the voice response.
    
    Args:
        input_file (str): Path to the input audio file (.wav)
        output_file (str): Path to save the output audio file
        model (str): OpenAI Realtime model to use
        voice (str): Voice to use for the response (alloy, shimmer, nova, etc.)
    """
    global recording_started, recorder, received_audio, dc, response_started, response_done
    
    # Validate the input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist")
        return False
        
    url = f"https://api.openai.com/v1/realtime?model={model}"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/sdp"
    }
    
    print(f"Sending voice request from file: {input_file}")
    print(f"Response will be saved to: {output_file}")
    
    # Create a temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up the recorder path
        recorder_path = os.path.join(temp_dir, "recorded")
        
        # Create a new RTCPeerConnection
        pc = RTCPeerConnection()
        
        # Create a data channel for sending/receiving events
        dc = pc.createDataChannel("events")
        
        # Create an audio track from the input file
        audio_track = AudioFileTrack(input_file)
        pc.addTrack(audio_track)
        
        # Set up audio recorder for the remote tracks but don't start yet
        recorder = MediaRecorder(output_file)
        
        # Handler for messages received on the data channel
        @dc.on("open")
        def on_open():
            print("Data channel opened - connection established")
            
            # Configure session with voice
            event = {
                "type": "session.update",
                "session": {
                    "voice": voice
                }
            }
            dc.send(json.dumps(event))
            
            # After a short delay, create a response to process the audio
            asyncio.create_task(send_response_create())
            
        @dc.on("message")
        def on_datachannel_message(message):
            global recording_started, received_audio, response_started, response_done
            try:
                event = json.loads(message)
                event_type = event.get("type", "")
                
                print(f"Received event: {event_type}")
                
                # Track important events
                if event_type == "response.audio.delta":
                    if not received_audio:
                        print("Receiving audio response...")
                        received_audio = True
                        # If we get audio but haven't started recording yet, start now
                        if not recording_started:
                            asyncio.create_task(start_recorder())
                
                elif event_type == "response.completed":
                    print("Response complete")
                    response_done = True
                    # Schedule a task to end the connection after a short delay
                    asyncio.create_task(schedule_shutdown())
                
                elif event_type == "session.created":
                    print(f"Session created with model: {model}")
                
                elif event_type == "response.started":
                    print("Response started - model is processing the request")
                    response_started = True
                    # Start recorder when response starts
                    if not recording_started:
                        asyncio.create_task(start_recorder())
                
                elif event_type == "response.done":
                    print("Response done")
                    response_done = True
                    # Important: If we get response.done without response.started, we need to handle that
                    if not response_started:
                        print("Note: Received response.done before response.started")
                        response_started = True
                        # Make sure we start recording if there's any content
                        if not recording_started:
                            asyncio.create_task(start_recorder())
                    # Schedule a task to end the connection after a short delay
                    asyncio.create_task(schedule_shutdown())
                
            except Exception as e:
                print(f"Error processing message: {e}")
        
        # When a track is received from the server, add it to the recorder
        @pc.on("track")
        async def on_track(track):
            print(f"Received track of kind: {track.kind}")
            if track.kind == "audio":
                # Add the track but don't start recording yet
                recorder.addTrack(track)
        
        # Create an offer to send to the server
        await pc.setLocalDescription(await pc.createOffer())
        
        # Send the offer to the server and get the answer
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=pc.localDescription.sdp) as response:
                if response.status not in [200, 201]:
                    error_text = await response.text()
                    print(f"Error connecting to OpenAI Realtime API: {response.status} - {error_text}")
                    return False
                
                # Get the SDP answer from the server
                sdp_answer = await response.text()
                
                # Set the remote description with the server's answer
                await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp_answer, type="answer"))
        
        async def send_response_create():
            # Wait a moment before sending response.create to ensure audio has started
            await asyncio.sleep(2)
            
            # Send instructions to treat the audio as a question to answer
            event = {
                "type": "response.create",
                "response": {
                    "instructions": "This is a user asking a question. Please listen carefully to the audio and provide a direct and helpful answer to their question."
                }
            }
            print("Sending audio and requesting response...")
            dc.send(json.dumps(event))
            
            # Start monitoring for response to actually begin
            asyncio.create_task(wait_for_response())
        
        async def start_recorder():
            global recording_started
            if not recording_started:
                print("Starting recorder now...")
                recording_started = True
                await recorder.start()
                print("Recorder started")
        
        async def wait_for_response():
            global recording_started, response_started, response_done
            # Wait for response to start with a timeout
            timeout = 15  # seconds - increased timeout
            start_time = asyncio.get_event_loop().time()
            
            while not response_started and not response_done:
                if asyncio.get_event_loop().time() - start_time > timeout:
                    print("Warning: Timed out waiting for response to start")
                    # Even if we time out, let's try to start recording anyway
                    # in case there's audio data coming but events were missed
                    if not recording_started:
                        await start_recorder()
                    break
                await asyncio.sleep(0.1)
            
            # Start recorder if we have a response but haven't started recording yet
            if (response_started or response_done) and not recording_started:
                await start_recorder()
        
        async def schedule_shutdown():
            # Give some time for final audio packets to arrive
            await asyncio.sleep(3)
            print("Closing connection...")
            
            # Close recorder and connection
            if recorder:
                try:
                    await recorder.stop()
                except Exception as e:
                    print(f"Error stopping recorder: {e}")
            
            await pc.close()
        
        # Run for a reasonable amount of time to capture the full interaction
        try:
            await asyncio.sleep(60)  # Timeout after 60 seconds if not completed
        except asyncio.CancelledError:
            pass
        
        # Ensure recorder is stopped
        if recorder:
            try:
                await recorder.stop()
            except Exception as e:
                print(f"Error stopping recorder during cleanup: {e}")
        
        # Close the connection
        await pc.close()
    
    print(f"Voice response saved to: {output_file}")
    return True

def process_voice_file(input_file, output_file="response.wav", model="gpt-4o-realtime-preview", voice="alloy"):
    """
    Main function to process a voice file and get a response from OpenAI Realtime API.
    
    Args:
        input_file (str): Path to the input audio file (.wav)
        output_file (str): Path to save the output audio file (default: response.wav)
        model (str): OpenAI Realtime model to use (default: gpt-4o-realtime-preview)
        voice (str): Voice to use for the response (default: alloy)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Reset global state
        global received_audio, recording_started, recorder, dc, response_started, response_done
        received_audio = False
        recording_started = False
        recorder = None
        dc = None
        response_started = False
        response_done = False
        
        # Run the asyncio event loop
        return asyncio.run(run_webrtc_client(input_file, output_file, model, voice))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

# Remove command-line parsing completely - only use function calls
if __name__ == "__main__":
    # This is just a demonstration of how to call the function
    # and can be removed or modified as needed
    process_voice_file("harvard.wav")

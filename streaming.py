import os
import json
import websocket
import _thread as thread
import time
import sys

# Get API key from environment variable instead of hardcoding
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY environment variable not set")
    print("Please set it using: export OPENAI_API_KEY='your_api_key'")
    exit(1)

url = "wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview"
headers = [
    "Authorization: Bearer " + OPENAI_API_KEY,
    "OpenAI-Beta: realtime=v1"
]

# Variable to accumulate the complete response
complete_response = ""

def on_open(ws):
    # print("Connected to server.")
    
    # Define function to send messages after connection is established
    def run(*args):
        try:
            # Send user message
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
            ws.send(json.dumps(event))
            # print("User message sent")
            
            # Request response
            event = {
                "type": "response.create",
                "response": {
                    "modalities": ["text"]
                }
            }
            ws.send(json.dumps(event))
            # print("Response requested")
            
            # Keep the main thread running for a while to receive responses
            time.sleep(30)
            
            # Close connection after receiving responses
            ws.close()
            # print("Connection closed")
        except Exception as e:
            print(f"Error: {e}")
            ws.close()
    
    # Start a new thread for sending messages
    thread.start_new_thread(run, ())

def on_message(ws, message):
    global complete_response
    try:
        server_event = json.loads(message)
        event_type = server_event.get("type", "")
        
        # Only print certain event types for debugging
        if event_type == "session.created":
            pass
            # print("Session created successfully")
        elif event_type == "conversation.item.created" and server_event.get("item", {}).get("role") == "user":
            # print("User message acknowledged by server")
            pass
        elif event_type == "response.created":
            # print("AI is generating a response...")
            complete_response = ""  # Reset the response buffer
        # Handle streaming text deltas - accumulate rather than display
        elif event_type == "response.text.delta":
            delta = server_event.get("delta", "")
            complete_response += delta  # Accumulate the response
        # When response is complete, display full response
        elif event_type == "response.done":
            print( complete_response)
            # print("\nResponse complete.")
            
        # For detailed debugging, uncomment this:
        # print(f"DEBUG: {event_type}")
            
    except Exception as e:
        print(f"Error handling message: {e}")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print(f"Connection closed: {close_status_code} - {close_msg}")

if __name__ == "__main__":
    # Initialize WebSocket with all handlers
    ws = websocket.WebSocketApp(
        url,
        header=headers,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    
    # Start WebSocket connection (blocking call)
    ws.run_forever()
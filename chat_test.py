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
# Flag to track when a response is complete
response_complete = True
# Flag to control the main loop
running = True

def on_open(ws):
    print("Connected to server. Type your questions (or 'exit' to quit):")
    
    # Define function to send messages after connection is established
    def run(*args):
        global running, response_complete
        
        try:
            while running:
                # Wait for previous response to complete before asking for new input
                if response_complete:
                    # Get user input
                    user_input = input("> ")
                    
                    # Check if user wants to exit
                    if user_input.lower() in ["exit", "quit"]:
                        print("Closing connection...")
                        running = False
                        ws.close()
                        break
                    
                    # Set flag to indicate waiting for response
                    response_complete = False
                    
                    # Send user message
                    event = {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": user_input,
                                }
                            ]
                        }
                    }
                    ws.send(json.dumps(event))
                    
                    # Request response
                    event = {
                        "type": "response.create",
                        "response": {
                            "modalities": ["text"]
                        }
                    }
                    ws.send(json.dumps(event))
                
                # Small sleep to prevent CPU overuse in the loop
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Error: {e}")
            ws.close()
    
    # Start a new thread for interaction
    thread.start_new_thread(run, ())

def on_message(ws, message):
    global complete_response, response_complete
    try:
        server_event = json.loads(message)
        event_type = server_event.get("type", "")
        
        if event_type == "response.created":
            complete_response = ""  # Reset the response buffer
            print("AI is thinking...")
        # Handle streaming text deltas - accumulate rather than display
        elif event_type == "response.text.delta":
            delta = server_event.get("delta", "")
            complete_response += delta  # Accumulate the response
            # Print the delta for real-time display
            print(delta, end="", flush=True)
        # When response is complete, display full response
        elif event_type == "response.done":
            print("\n")  # Add a new line after response
            response_complete = True  # Mark response as complete
            
    except Exception as e:
        print(f"Error handling message: {e}")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print(f"Connection closed: {close_status_code} - {close_msg}")
    sys.exit(0)

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
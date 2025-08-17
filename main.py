import asyncio
import socketio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import the refactored processing function
from base import process_video_stream

# --- App and WebSocket Setup ---
app = FastAPI()
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins="*")
sio_app = socketio.ASGIApp(sio)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Model Configuration ---
MODEL_CONFIG = {
    'det_weight': "./weights/det_10g.onnx",
    'rec_weight': "./weights/w600k_r50.onnx",
    'gun_weight': "./best.pt",
    'faces_dir': "./faces",
    'similarity_thresh': 0.5,
    'gun_conf_thresh': 0.7,
    'camera_id': 0
}

# --- Background Task for Video Processing ---
async def run_video_processing():
    """The background task that runs the main detection loop."""
    print("Starting model processing in the background...")
    # The process_video_stream is a generator, so we iterate through its yielded results
    for frame_data in process_video_stream(MODEL_CONFIG):
        # Emit the data to all connected clients
        await sio.emit('update', frame_data)
        # Yield control to the event loop briefly
        await asyncio.sleep(0.01)

@app.on_event("startup")
async def startup_event():
    """On server startup, run the video processing in a background task."""
    sio.start_background_task(run_video_processing)

# --- WebSocket Events ---
@sio.event
async def connect(sid, environ):
    print(f"Dashboard connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"Dashboard disconnected: {sid}")

# Mount the Socket.IO app to the main FastAPI app
app.mount('/', sio_app)

if __name__ == '__main__':
    uvicorn.run("main:app", host='127.0.0.1', port=8000, reload=True)

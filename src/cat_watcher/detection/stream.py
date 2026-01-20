"""RTSP stream reader for video processing.

Provides reliable frame reading from RTSP streams with automatic reconnection,
frame rate control, and async iteration support.
"""

import asyncio
import os
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from queue import Empty, Full, Queue
from typing import Any

# Suppress FFmpeg warnings (SEI truncation, etc.) before importing cv2
# These are harmless decoder messages that clutter logs
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")  # AV_LOG_QUIET

import cv2
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class StreamState(str, Enum):
    """State of the stream reader."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class StreamStats:
    """Statistics for stream reading."""

    frames_read: int = 0
    frames_yielded: int = 0
    frames_dropped: int = 0
    reconnects: int = 0
    errors: int = 0
    last_frame_time: float = 0.0
    start_time: float = field(default_factory=time.time)

    @property
    def uptime(self) -> float:
        """Get uptime in seconds."""
        return time.time() - self.start_time

    @property
    def effective_fps(self) -> float:
        """Get effective frames per second yielded."""
        if self.uptime > 0:
            return self.frames_yielded / self.uptime
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frames_read": self.frames_read,
            "frames_yielded": self.frames_yielded,
            "frames_dropped": self.frames_dropped,
            "reconnects": self.reconnects,
            "errors": self.errors,
            "uptime_seconds": round(self.uptime, 1),
            "effective_fps": round(self.effective_fps, 2),
        }


class StreamReader:
    """Reads frames from an RTSP stream with automatic reconnection.

    This class handles:
    - Connecting to RTSP streams
    - Reading frames at a target FPS
    - Automatic reconnection on failure
    - Async iteration for frame processing

    Example:
        ```python
        reader = StreamReader("rtsp://user:pass@camera/stream1", target_fps=5.0)
        await reader.start()

        async for timestamp, frame in reader.frames():
            # Process frame (BGR numpy array)
            print(f"Frame at {timestamp}: {frame.shape}")

        await reader.stop()
        ```
    """

    def __init__(
        self,
        url: str,
        target_fps: float = 5.0,
        reconnect_delay: float = 5.0,
        max_reconnect_delay: float = 60.0,
        queue_size: int = 10,
        connection_timeout: float = 30.0,
    ):
        """Initialize stream reader.

        Args:
            url: RTSP stream URL
            target_fps: Target frames per second to yield (skip frames if source is faster)
            reconnect_delay: Initial delay between reconnection attempts (seconds)
            max_reconnect_delay: Maximum delay between reconnection attempts (seconds)
            queue_size: Maximum frames to buffer
            connection_timeout: Timeout for initial connection (seconds)
        """
        self.url = url
        self.target_fps = target_fps
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.queue_size = queue_size
        self.connection_timeout = connection_timeout

        # Internal state
        self._state = StreamState.DISCONNECTED
        self._stop_event = threading.Event()
        self._frame_queue: Queue[tuple[float, np.ndarray]] = Queue(maxsize=queue_size)
        self._capture_thread: threading.Thread | None = None
        self._stats = StreamStats()
        self._frame_interval = 1.0 / target_fps
        self._last_yield_time = 0.0

        # Stream info (populated after connection)
        self._width: int = 0
        self._height: int = 0
        self._source_fps: float = 0.0

        # Mask credentials in URL for logging
        self._masked_url = self._mask_url(url)

    @staticmethod
    def _mask_url(url: str) -> str:
        """Mask credentials in URL for safe logging."""
        if "@" in url:
            # rtsp://user:pass@host/path -> rtsp://***:***@host/path
            prefix, rest = url.split("@", 1)
            if "://" in prefix:
                scheme, _ = prefix.split("://", 1)
                return f"{scheme}://***:***@{rest}"
        return url

    @property
    def state(self) -> StreamState:
        """Get current stream state."""
        return self._state

    @property
    def stats(self) -> StreamStats:
        """Get stream statistics."""
        return self._stats

    @property
    def is_connected(self) -> bool:
        """Check if stream is connected."""
        return self._state == StreamState.CONNECTED

    @property
    def resolution(self) -> tuple[int, int]:
        """Get stream resolution (width, height)."""
        return (self._width, self._height)

    @property
    def source_fps(self) -> float:
        """Get source stream FPS."""
        return self._source_fps

    async def start(self) -> None:
        """Start reading from stream.

        Raises:
            RuntimeError: If already started
            ConnectionError: If initial connection fails
        """
        if self._capture_thread is not None and self._capture_thread.is_alive():
            raise RuntimeError("Stream reader already started")

        log = logger.bind(url=self._masked_url)
        log.info("Starting stream reader")

        self._stop_event.clear()
        self._stats = StreamStats()
        self._state = StreamState.CONNECTING

        # Start capture thread
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name=f"StreamReader-{self._masked_url[:30]}",
            daemon=True,
        )
        self._capture_thread.start()

        # Wait for initial connection
        start_time = time.time()
        while self._state == StreamState.CONNECTING:
            if time.time() - start_time > self.connection_timeout:
                self._stop_event.set()
                self._state = StreamState.ERROR
                raise ConnectionError(
                    f"Timeout connecting to stream: {self._masked_url}"
                )
            await asyncio.sleep(0.1)

        if self._state == StreamState.ERROR:
            raise ConnectionError(f"Failed to connect to stream: {self._masked_url}")

        log.info(
            "Stream reader started",
            resolution=f"{self._width}x{self._height}",
            source_fps=self._source_fps,
            target_fps=self.target_fps,
        )

    async def stop(self) -> None:
        """Stop reading from stream."""
        log = logger.bind(url=self._masked_url)
        log.info("Stopping stream reader")

        self._stop_event.set()
        self._state = StreamState.STOPPED

        # Wait for capture thread to finish
        if self._capture_thread is not None and self._capture_thread.is_alive():
            # Give it a moment to clean up
            await asyncio.sleep(0.5)
            if self._capture_thread.is_alive():
                log.warning("Capture thread still running after stop")

        # Clear queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except Empty:
                break

        log.info("Stream reader stopped", stats=self._stats.to_dict())

    async def frames(self) -> AsyncIterator[tuple[float, np.ndarray]]:
        """Async iterator yielding (timestamp, frame) tuples.

        Yields:
            Tuple of (unix_timestamp, frame) where frame is a BGR numpy array

        Raises:
            RuntimeError: If stream not started or has stopped
        """
        if self._state == StreamState.DISCONNECTED:
            raise RuntimeError("Stream not started. Call start() first.")

        while not self._stop_event.is_set():
            # Check state
            if self._state == StreamState.ERROR:
                raise RuntimeError(f"Stream error: {self._masked_url}")

            if self._state == StreamState.STOPPED:
                return

            # Try to get a frame
            try:
                timestamp, frame = self._frame_queue.get_nowait()

                # Rate limiting - skip if too soon since last yield
                now = time.time()
                if now - self._last_yield_time < self._frame_interval * 0.9:
                    # Too soon, drop this frame
                    self._stats.frames_dropped += 1
                    continue

                self._last_yield_time = now
                self._stats.frames_yielded += 1
                yield timestamp, frame

            except Empty:
                # No frame available, wait a bit
                await asyncio.sleep(0.01)

    def _capture_loop(self) -> None:
        """Main capture loop running in separate thread."""
        log = logger.bind(url=self._masked_url)
        current_delay = self.reconnect_delay

        while not self._stop_event.is_set():
            cap = None
            try:
                # Connect to stream
                log.debug("Connecting to stream")
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

                if not cap.isOpened():
                    raise ConnectionError("Failed to open stream")

                # Get stream properties
                self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self._source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

                # Set buffer size to reduce latency
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                self._state = StreamState.CONNECTED
                current_delay = self.reconnect_delay  # Reset delay on success

                log.info(
                    "Connected to stream",
                    resolution=f"{self._width}x{self._height}",
                    fps=self._source_fps,
                )

                # Read frames
                consecutive_failures = 0
                while not self._stop_event.is_set():
                    ret, frame = cap.read()

                    if not ret or frame is None:
                        consecutive_failures += 1
                        if consecutive_failures > 30:
                            log.warning("Too many consecutive read failures")
                            break
                        time.sleep(0.01)
                        continue

                    consecutive_failures = 0
                    self._stats.frames_read += 1
                    timestamp = time.time()
                    self._stats.last_frame_time = timestamp

                    # Put frame in queue (non-blocking)
                    try:
                        self._frame_queue.put_nowait((timestamp, frame))
                    except Full:
                        # Queue full, drop oldest frame
                        try:
                            self._frame_queue.get_nowait()
                            self._frame_queue.put_nowait((timestamp, frame))
                        except (Empty, Full):
                            pass
                        self._stats.frames_dropped += 1

            except Exception as e:
                self._stats.errors += 1
                log.error("Stream error", error=str(e))

                if self._state == StreamState.CONNECTING:
                    # Initial connection failed
                    self._state = StreamState.ERROR
                    return

            finally:
                if cap is not None:
                    cap.release()

            # Reconnect logic
            if self._stop_event.is_set():
                break

            self._state = StreamState.RECONNECTING
            self._stats.reconnects += 1
            log.info(f"Reconnecting in {current_delay:.1f}s")

            # Wait with exponential backoff
            self._stop_event.wait(current_delay)
            current_delay = min(current_delay * 1.5, self.max_reconnect_delay)

        log.debug("Capture loop exiting")


async def get_camera_rtsp_url(
    frigate_url: str,
    camera: str,
    rtsp_username: str | None = None,
    rtsp_password: str | None = None,
) -> str:
    """Get RTSP URL for a camera from Frigate config.

    Args:
        frigate_url: Frigate API URL (e.g., http://192.168.50.36:5000)
        camera: Camera name
        rtsp_username: RTSP username (to replace masked credentials)
        rtsp_password: RTSP password (to replace masked credentials)

    Returns:
        RTSP URL for the camera

    Raises:
        ValueError: If camera not found
        ConnectionError: If Frigate unreachable
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{frigate_url}/api/config")
            response.raise_for_status()
            config = response.json()

            cameras = config.get("cameras", {})
            if camera not in cameras:
                available = list(cameras.keys())
                raise ValueError(
                    f"Camera '{camera}' not found. Available: {available}"
                )

            cam_config = cameras[camera]
            inputs = cam_config.get("ffmpeg", {}).get("inputs", [])

            if not inputs:
                raise ValueError(f"No inputs configured for camera '{camera}'")

            # Get first input with detect or record role
            url = None
            for inp in inputs:
                roles = inp.get("roles", [])
                if "detect" in roles or "record" in roles:
                    url = inp["path"]
                    break

            if url is None:
                # Fall back to first input
                url = inputs[0]["path"]

            # Replace masked credentials (Frigate uses *:* for masked creds)
            if rtsp_username and rtsp_password:
                # Handle formats: rtsp://*:*@host or rtsp://user:pass@host
                if "@" in url:
                    # Extract scheme and rest
                    scheme_end = url.find("://") + 3
                    scheme = url[:scheme_end]
                    rest = url[scheme_end:]
                    
                    # Split on @ to get host part
                    _, host_part = rest.split("@", 1)
                    
                    # Rebuild URL with real credentials
                    url = f"{scheme}{rtsp_username}:{rtsp_password}@{host_part}"

            return url

    except httpx.RequestError as e:
        raise ConnectionError(f"Failed to connect to Frigate: {e}") from e


async def list_cameras(frigate_url: str) -> list[dict[str, Any]]:
    """List available cameras from Frigate.

    Args:
        frigate_url: Frigate API URL

    Returns:
        List of camera info dicts with name, resolution, fps
    """
    import httpx

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{frigate_url}/api/config")
        response.raise_for_status()
        config = response.json()

        cameras = []
        for name, cam_config in config.get("cameras", {}).items():
            detect = cam_config.get("detect", {})
            cameras.append({
                "name": name,
                "width": detect.get("width", 0),
                "height": detect.get("height", 0),
                "fps": detect.get("fps", 5),
                "enabled": cam_config.get("enabled", True),
            })

        return cameras

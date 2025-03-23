from uuid import UUID
import asyncio
from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect
from starlette.websockets import WebSocketState
import logging
from typing import Any
from util import ParamsModel

Connections = dict[UUID, dict[str, WebSocket | asyncio.Queue]]


class ServerFullException(Exception):
    """Exception raised when the server is full."""

    pass


class ConnectionManager:
    def __init__(self):
        self.active_connections: Connections = {}

    async def connect(
        self, user_id: UUID, websocket: WebSocket, max_queue_size: int = 0
    ):
        await websocket.accept()
        user_count = self.get_user_count()
        print(f"User count: {user_count}")
        if max_queue_size > 0 and user_count >= max_queue_size:
            print("Server is full")
            await websocket.send_json({"status": "error", "message": "Server is full"})
            await websocket.close()
            raise ServerFullException("Server is full")
        print(f"New user connected: {user_id}")
        self.active_connections[user_id] = {
            "websocket": websocket,
            "queue": asyncio.Queue(),
        }
        await websocket.send_json(
            {"status": "connected", "message": "Connected"},
        )
        await websocket.send_json({"status": "wait"})
        await websocket.send_json({"status": "send_frame"})

    def check_user(self, user_id: UUID) -> bool:
        return user_id in self.active_connections

    async def update_data(self, user_id: UUID, new_data: ParamsModel):
        user_session = self.active_connections.get(user_id)
        if user_session:
            queue = user_session["queue"]
            await queue.put(new_data)

    async def get_latest_data(self, user_id: UUID) -> ParamsModel | None:
        user_session = self.active_connections.get(user_id)
        if user_session:
            queue = user_session["queue"]
            try:
                return await queue.get()
            except asyncio.QueueEmpty:
                return None
        return None

    def delete_user(self, user_id: UUID):
        user_session = self.active_connections.pop(user_id, None)
        if user_session:
            queue = user_session["queue"]
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    continue

    def get_user_count(self) -> int:
        return len(self.active_connections)

    def get_websocket(self, user_id: UUID) -> WebSocket | None:
        user_session = self.active_connections.get(user_id)
        if user_session:
            websocket = user_session["websocket"]
            # Both client_state and application_state should be checked
            # to ensure the websocket is fully connected and not closing
            if (websocket.client_state == WebSocketState.CONNECTED and 
                websocket.application_state == WebSocketState.CONNECTED):
                return user_session["websocket"]
        return None

    async def disconnect(self, user_id: UUID):
        # First check if user is in active connections
        if user_id not in self.active_connections:
            return
            
        # Get the websocket directly from active_connections to avoid get_websocket validation
        user_session = self.active_connections.get(user_id)
        if user_session and "websocket" in user_session:
            websocket = user_session["websocket"]
            try:
                # Only attempt close if not already closed
                if (websocket.client_state != WebSocketState.DISCONNECTED and 
                    websocket.application_state != WebSocketState.DISCONNECTED):
                    await websocket.close()
            except Exception as e:
                logging.error(f"Error closing websocket for {user_id}: {e}")
                
        # Always delete the user to ensure cleanup
        self.delete_user(user_id)

    async def send_json(self, user_id: UUID, data: dict):
        try:
            websocket = self.get_websocket(user_id)
            if websocket:
                try:
                    await websocket.send_json(data)
                except RuntimeError as e:
                    error_msg = str(e)
                    if any(err in error_msg for err in [
                        "WebSocket is not connected", 
                        "Cannot call \"send\" once a close message has been sent",
                        "Cannot call \"receive\" once a close message has been sent",
                        "WebSocket is disconnected"]):
                        # The websocket was disconnected or is closing
                        logging.info(f"WebSocket disconnected/closing for user {user_id}: {error_msg}")
                        await self.disconnect(user_id)
                    else:
                        logging.error(f"Runtime error in send_json: {e}")
        except WebSocketDisconnect as disconnect_error:
            # Handle websocket disconnection event
            code = disconnect_error.code
            if code == 1006:  # ABNORMAL_CLOSURE
                logging.info(f"WebSocket abnormally closed for user {user_id} during send: Connection was closed without a proper close handshake")
            else:
                logging.info(f"WebSocket disconnected for user {user_id} with code {code} during send: {disconnect_error.reason}")
            
            # Always disconnect the user
            if user_id in self.active_connections:
                await self.disconnect(user_id)
        except Exception as e:
            logging.error(f"Error: Send json: {e}")
            # If any send fails, ensure the user gets removed to prevent further errors
            if user_id in self.active_connections:
                await self.disconnect(user_id)

    async def receive_json(self, user_id: UUID) -> dict | None:
        try:
            websocket = self.get_websocket(user_id)
            if websocket:
                try:
                    # Receive the raw message and handle JSON parsing manually for better error handling
                    try:
                        data = await websocket.receive_json()
                        # Verify it's a dictionary
                        if not isinstance(data, dict):
                            logging.error(f"Expected dict but received {type(data)} from user {user_id}: {data}")
                            return None
                        return data
                    except ValueError as json_err:
                        # Specific handling for JSON parsing errors
                        logging.error(f"JSON parsing error for user {user_id}: {json_err}")
                        return None
                except RuntimeError as e:
                    error_msg = str(e)
                    if any(err in error_msg for err in [
                        "WebSocket is not connected", 
                        "Cannot call \"send\" once a close message has been sent",
                        "Cannot call \"receive\" once a close message has been sent",
                        "WebSocket is disconnected"]):
                        # The websocket was disconnected or closing
                        logging.info(f"WebSocket disconnected/closing for user {user_id}: {error_msg}")
                        await self.disconnect(user_id)
                    else:
                        logging.error(f"Runtime error in receive_json: {e}")
                    return None
            return None
        except WebSocketDisconnect as disconnect_error:
            # Handle websocket disconnection event (this is a clean, expected path)
            code = disconnect_error.code
            if code == 1006:  # ABNORMAL_CLOSURE
                logging.info(f"WebSocket abnormally closed for user {user_id}: Connection was closed without a proper close handshake")
            else:
                logging.info(f"WebSocket disconnected for user {user_id} with code {code}: {disconnect_error.reason}")
            
            # Always disconnect the user
            if user_id in self.active_connections:
                await self.disconnect(user_id)
            return None
        except Exception as e:
            logging.error(f"Error: Receive json: {e}")
            # Ensure disconnection on any exception
            if user_id in self.active_connections:
                await self.disconnect(user_id)
            return None

    async def receive_bytes(self, user_id: UUID) -> bytes | None:
        try:
            websocket = self.get_websocket(user_id)
            if websocket:
                try:
                    return await websocket.receive_bytes()
                except RuntimeError as e:
                    error_msg = str(e)
                    if any(err in error_msg for err in [
                        "WebSocket is not connected", 
                        "Cannot call \"send\" once a close message has been sent",
                        "Cannot call \"receive\" once a close message has been sent",
                        "WebSocket is disconnected"]):
                        # The websocket was disconnected or closing
                        logging.info(f"WebSocket disconnected/closing for user {user_id}: {error_msg}")
                        await self.disconnect(user_id)
                    else:
                        logging.error(f"Runtime error in receive_bytes: {e}")
                    return None
            return None
        except WebSocketDisconnect as disconnect_error:
            # Handle websocket disconnection event (this is a clean, expected path)
            code = disconnect_error.code
            if code == 1006:  # ABNORMAL_CLOSURE
                logging.info(f"WebSocket abnormally closed for user {user_id}: Connection was closed without a proper close handshake")
            else:
                logging.info(f"WebSocket disconnected for user {user_id} with code {code}: {disconnect_error.reason}")
            
            # Always disconnect the user
            if user_id in self.active_connections:
                await self.disconnect(user_id)
            return None
        except Exception as e:
            logging.error(f"Error: Receive bytes: {e}")
            # Ensure disconnection on any exception
            if user_id in self.active_connections:
                await self.disconnect(user_id)
            return None

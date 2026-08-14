from typing import Dict, Optional, Tuple
import asyncio
from loguru import logger
from collections import defaultdict

from .utils.turn_trace import record_turn_event


class MessageHandler:
    def __init__(self):
        self._response_events: Dict[
            str, Dict[Tuple[str, Optional[str]], asyncio.Event]
        ] = defaultdict(dict)
        self._response_data: Dict[str, Dict[Tuple[str, Optional[str]], dict]] = (
            defaultdict(dict)
        )

    async def wait_for_response(
        self,
        client_uid: str,
        response_type: str,
        request_id: str | None = None,
        timeout: float | None = None,
    ) -> Optional[dict]:
        """
        Wait for a response of specific type and optional request_id from a client.

        Args:
            client_uid: Client identifier
            response_type: Type of response to wait for
            request_id: Optional identifier for the specific request
            timeout: Optional timeout in seconds. If None, wait indefinitely

        Returns:
            Optional[dict]: Response data if received, None if timeout
        """
        waiter = self.prepare_response_wait(
            client_uid=client_uid,
            response_type=response_type,
            request_id=request_id,
            timeout=timeout,
        )
        return await waiter

    def prepare_response_wait(
        self,
        client_uid: str,
        response_type: str,
        request_id: str | None = None,
        timeout: float | None = None,
    ):
        """Register a response wait before sending the request that triggers it."""
        event = asyncio.Event()
        response_key = (response_type, request_id)
        self._response_events[client_uid][response_key] = event
        record_turn_event(
            request_id,
            "message_handler",
            "response_wait_registered",
            client_uid=client_uid,
            response_type=response_type,
            timeout=timeout,
        )

        return self._wait_for_registered_response(
            client_uid=client_uid,
            response_type=response_type,
            request_id=request_id,
            timeout=timeout,
            event=event,
        )

    async def _wait_for_registered_response(
        self,
        client_uid: str,
        response_type: str,
        request_id: str | None,
        timeout: float | None,
        event: asyncio.Event,
    ) -> Optional[dict]:
        response_key = (response_type, request_id)
        try:
            if timeout is not None:
                # Wait with timeout
                await asyncio.wait_for(event.wait(), timeout)
            else:
                # Wait indefinitely
                await event.wait()

            response = self._response_data[client_uid].pop(response_key, None)
            record_turn_event(
                request_id,
                "message_handler",
                "response_wait_completed",
                client_uid=client_uid,
                response_type=response_type,
                received=bool(response),
            )
            return response
        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout waiting for {response_type} (ID: {request_id}) from {client_uid}"
            )
            record_turn_event(
                request_id,
                "message_handler",
                "response_wait_timeout",
                client_uid=client_uid,
                response_type=response_type,
                timeout=timeout,
            )
            return None
        finally:
            self._response_events[client_uid].pop(response_key, None)

    def handle_message(self, client_uid: str, message: dict) -> bool:
        """
        Process an incoming message, potentially matching a response event waiting.

        Args:
            client_uid: Client identifier
            message: Message data dictionary, expected to contain 'type' and optionally 'request_id' or 'turn_id'

        Returns:
            bool: True when this message matched a pending response wait.
        """
        msg_type = message.get("type")
        request_id = message.get("request_id") or message.get("turn_id")
        if not msg_type:
            return False

        response_key = (msg_type, request_id)
        event = self._response_events.get(client_uid, {}).get(response_key)
        if not event:
            return False
        logger.info(f"message {message} for client_uid {client_uid} get event successed")
        if event.is_set():
            record_turn_event(
                request_id,
                "message_handler",
                "duplicate_response_ignored",
                client_uid=client_uid,
                response_type=msg_type,
            )
            logger.debug(
                "Ignoring duplicate {} response from {}",
                msg_type,
                client_uid,
            )
            return True

        self._response_data[client_uid][response_key] = message
        event.set()
        record_turn_event(
            request_id,
            "message_handler",
            "response_matched",
            client_uid=client_uid,
            response_type=msg_type,
        )
        return True

    def cleanup_client(self, client_uid: str) -> None:
        """
        Cleanup all events and cached data for a given client.

        Args:
            client_uid: Client identifier
        """
        if client_uid in self._response_events:
            for event in self._response_events[client_uid].values():
                event.set()
            self._response_events.pop(client_uid)
            self._response_data.pop(client_uid, None)


message_handler = MessageHandler()

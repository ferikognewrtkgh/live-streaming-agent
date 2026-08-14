"""
Open-LLM-VTuber Server
========================
This module contains the WebSocket server for Open-LLM-VTuber, which handles
the WebSocket connections, serves static files, and manages the web tool.
It uses FastAPI for the server and Starlette for static file serving.
"""

import os
import shutil

from loguru import logger

from .barrage_adapter import start_barrage_adapter, stop_barrage_adapter, BarrageConfig
from .orchestrator import start_orchestrator, stop_orchestrator
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response
from starlette.staticfiles import StaticFiles as StarletteStaticFiles

from .routes import init_client_ws_route, init_webtool_routes, init_proxy_route
from .service_context import ServiceContext
from .chat_history_manager import archive_all_working_histories
from .config_manager.utils import Config
from .resource_paths import AVATAR_ROOT, BACKGROUND_ROOT, LIVE2D_MODELS_ROOT
from .performance_metrics import configure_performance_storage
from .project_model_config import ProjectModelConfigManager


# Create a custom StaticFiles class that adds CORS headers
class CORSStaticFiles(StarletteStaticFiles):
    """
    Static files handler that adds CORS headers to all responses.
    Needed because Starlette StaticFiles might bypass standard middleware.
    """

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)

        # Add CORS headers to all responses
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"

        if path.endswith(".js"):
            response.headers["Content-Type"] = "application/javascript"

        return response


class AvatarStaticFiles(CORSStaticFiles):
    """
    Avatar files handler with security restrictions and CORS headers
    """

    async def get_response(self, path: str, scope):
        allowed_extensions = (".jpg", ".jpeg", ".png", ".gif", ".svg")
        if not any(path.lower().endswith(ext) for ext in allowed_extensions):
            return Response("Forbidden file type", status_code=403)
        response = await super().get_response(path, scope)
        return response


class WebSocketServer:
    """
    API server for Open-LLM-VTuber. This contains the websocket endpoint for the client, hosts the web tool, and serves static files.

    Creates and configures a FastAPI app, registers all routes
    (WebSocket, web tools, proxy) and mounts static assets with CORS.

    Args:
        config (Config): Application configuration containing system settings.
        default_context_cache (ServiceContext, optional):
            Pre閳ユ吔nitialized service context for sessions' service context to reference to.
            **If omitted, `initialize()` method needs to be called to load service context.**

    Notes:
        - If default_context_cache is omitted, call `await initialize()` to load service context cache.
        - Use `clean_cache()` to clear and recreate the local cache directory.
    """

    def __init__(self, config: Config, default_context_cache: ServiceContext = None):
        self.app = FastAPI(title="Open-LLM-VTuber Server")  # Added title for clarity
        self.config = config
        self.default_context_cache = (
            default_context_cache or ServiceContext()
        )  # Use provided context or initialize a new empty one waiting to be loaded
        # It will be populated during the initialize method call

        # Add global CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Include routes, passing the context instance
        # The context will be populated during the initialize step

        client_ws_router, self.ws_handler = init_client_ws_route(
            default_context_cache=self.default_context_cache
        )
        self.app.include_router(client_ws_router)

        self.app.include_router(
            init_webtool_routes(default_context_cache=self.default_context_cache),
        )

        # Initialize and include proxy routes if proxy is enabled
        system_config = config.system_config
        if hasattr(system_config, "enable_proxy") and system_config.enable_proxy:
            # Construct the server URL for the proxy
            host = system_config.host
            port = system_config.port
            server_url = f"ws://{host}:{port}/client-ws"
            self.app.include_router(
                init_proxy_route(server_url=server_url),
            )

        # Mount cache directory first (to ensure audio file access)
        if not os.path.exists("cache"):
            os.makedirs("cache")
        self.app.mount(
            "/cache",
            CORSStaticFiles(directory="cache"),
            name="cache",
        )

        # Mount static files with CORS-enabled handlers
        self.app.mount(
            "/live2d-models",
            CORSStaticFiles(directory=str(LIVE2D_MODELS_ROOT)),
            name="live2d-models",
        )
        self.app.mount(
            "/bg",
            CORSStaticFiles(directory=str(BACKGROUND_ROOT)),
            name="backgrounds",
        )
        self.app.mount(
            "/avatars",
            AvatarStaticFiles(directory=str(AVATAR_ROOT)),
            name="avatars",
        )

        # Mount web tool directory separately from frontend
        self.app.mount(
            "/web-tool",
            CORSStaticFiles(directory="web_tool", html=True),
            name="web_tool",
        )


       
        # ========== 瀵懓绠烽柅鍌炲帳閸?==========
        @self.app.on_event("startup")
        async def _start_barrage():
            # 启动 Orchestrator 调度中心
            self._orchestrator = await start_orchestrator(self.ws_handler)

            barrage_config = BarrageConfig(
                ws_url="ws://127.0.0.1:8888",
                consume_interval=5.0,
                min_content_length=2,
                gift_trigger_enabled=True,
                gift_trigger_min_diamonds=1,
                gift_max_consecutive=2,
                gift_dedup_window=30.0,
                semantic_dedup_window=180.0,
                semantic_dedup_threshold=0.55,
                response_rate_limit_count=3,
                response_rate_limit_window=300.0,
                stale_message_max_age=10.0,
                stale_warn_seconds=30.0,
                stale_force_reconnect_seconds=60.0,
                grab_exe_path="",
                grab_process_name="WssBarrageServer",
                grab_auto_restart=False,
                grab_restart_on_stale=False,
                grab_stale_restart_seconds=6.0,
                grab_restart_cooldown_seconds=6.0,
                grab_startup_wait_seconds=5.0,
            )
            logger.info(
                "[barrage] DouyinBarrageGrab process auto-start disabled; "
                "expecting an existing service at {}",
                barrage_config.ws_url,
            )
            await start_barrage_adapter(self.ws_handler, barrage_config)

            # 初始化 VTuber 模式状态机
            from .vtuber_state_machine import init_vtuber_state_machine
            self._vtuber_sm = init_vtuber_state_machine(
                ws_handler=self.ws_handler,
            )

        @self.app.on_event("shutdown")
        async def _stop_barrage():
            await stop_barrage_adapter()
            await stop_orchestrator()
            archived_paths = archive_all_working_histories()
            if archived_paths:
                logger.info(
                    f"Archived {len(archived_paths)} working history files on shutdown."
                )

    async def initialize(self):
        """Asynchronously load the service context from config.
        Calling this function is needed if default_context_cache was not provided to the constructor."""
        await self.default_context_cache.load_from_config(self.config)
        self.ws_handler.project_model_manager = ProjectModelConfigManager(self.config)
        try:
            await self.ws_handler.apply_project_model_config()
        except Exception:
            logger.exception(
                "Failed to apply saved project model config; keeping conf.yaml model."
            )
        await configure_performance_storage(
            self.config.knowledge_config,
            self.default_context_cache.knowledge_runtime,
        )

    @staticmethod
    def clean_cache():
        """Clean the cache directory by removing and recreating it."""
        cache_dir = "cache"
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir)

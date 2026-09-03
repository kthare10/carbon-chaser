"""FastAPI dashboard: live state plus the one booth control that is real.

There is no spike-injection endpoint: injecting a fake grid event would
put fabricated values into a run whose whole point is measured data. The
remaining control forces a migration, which is a real action with real
measured cost.
"""

import os

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .engine import Engine

STATIC = os.path.join(os.path.dirname(__file__), "static")


class MigrateRequest(BaseModel):
    site: str


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="carbon-chaser")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC, "index.html"))

    @app.get("/api/state")
    def state():
        return JSONResponse(engine.state())

    @app.post("/api/migrate")
    def migrate(req: MigrateRequest):
        """Booth button: force a migration regardless of policy."""
        return {"ok": engine.force_migration(req.site)}

    return app

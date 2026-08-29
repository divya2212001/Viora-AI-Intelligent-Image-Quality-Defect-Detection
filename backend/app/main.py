from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings

from .database import (
    check_database_connection,
    close_database_connection,
)

from .routes.analysis import (
    router as analysis_router,
)

from .routes.health import (
    router as health_router,
)

from .routes.prediction import (
    router as prediction_router,
)
from .routes.model_info import router as model_info_router



# APPLICATION LIFESPAN
@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    """
    Application startup/shutdown lifecycle.
    """


    # Startup
    settings.UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not check_database_connection():

        print(
            "WARNING: MongoDB is not reachable."
        )

    else:

        print(
            "MongoDB connection successful."
        )

    print(
        "Application startup complete."
    )

    yield

    # Shutdown
    close_database_connection()

    print(
        "Application shutdown complete."
    )



# FASTAPI APPLICATION
app = FastAPI(

    title=settings.APP_NAME,

    version=settings.APP_VERSION,

    description=(
        "AI-powered image quality and "
        "visual defect detection API "
        "using a Hybrid CNN + Computer "
        "Vision model."
    ),

    lifespan=lifespan,
)



# CORS
app.add_middleware(

    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],

    allow_credentials=True,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ],
)



# STATIC UPLOADS
app.mount(

    "/uploads",

    StaticFiles(
        directory=str(
            settings.UPLOAD_DIR
        )
    ),

    name="uploads",
)



# ROOT
@app.get("/")
def root():

    return {

        "application":
            settings.APP_NAME,

        "status":
            "running",

        "version":
            settings.APP_VERSION,

        "docs":
            "/docs",

        "health":
            "/health",
    }



# ROUTES
app.include_router(
    health_router
)

app.include_router(
    analysis_router
)

app.include_router(
    prediction_router
)

app.include_router(model_info_router)

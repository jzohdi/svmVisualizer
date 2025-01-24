from fastapi import FastAPI
from fastapi_admin.app import app as admin_app
from fastapi_admin.providers.login import UsernamePasswordProvider
from fastapi_admin.template import templates
from .models import SvmResult
from .database import engine, AsyncSessionLocal

# Admin app initialization
async def init_admin(app: FastAPI):
    # Initialize FastAPI Admin
    await admin_app.init(
        admin_app=app,
        engine=engine,
        sessionmaker=AsyncSessionLocal,
        templates=templates,
        login_providers=[
            UsernamePasswordProvider(
                login_template="login.html",
                username_field="username",
                password_field="password",
            )
        ],
    )
    # Add your models to the admin panel
    await admin_app.register_model(SvmResult, display_fields=["id", "method", "result"])

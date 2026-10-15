from .app import app, create_app
import uvicorn


if __name__ == "__main__":
    # Reuse the import-time app (models load once); without COGNIMEM_JWT_SECRET it is None and create_app explains why.
    uvicorn.run(app if app is not None else create_app(), host="0.0.0.0", port=8000)

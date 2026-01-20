# FastAPI Quick Guide

## What is FastAPI?

FastAPI is a modern, high-performance web framework for building APIs with Python 3.7+ based on standard Python type hints. It's one of the fastest Python frameworks available, on par with NodeJS and Go.

## Key Features

- **Fast**: Very high performance, on par with NodeJS and Go (thanks to Starlette and Pydantic)
- **Fast to code**: Increase development speed by 200-300%
- **Fewer bugs**: Reduce human errors by about 40%
- **Intuitive**: Great editor support with autocompletion
- **Easy**: Designed to be easy to use and learn
- **Short**: Minimize code duplication
- **Robust**: Production-ready code with automatic interactive documentation
- **Standards-based**: Based on OpenAPI and JSON Schema

## Installation

```bash
pip install fastapi
pip install "uvicorn[standard]"  # ASGI server
```

## Basic Example

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

Run the server:
```bash
uvicorn main:app --reload
```

Access:
- API: http://127.0.0.1:8000
- Interactive docs: http://127.0.0.1:8000/docs
- Alternative docs: http://127.0.0.1:8000/redoc

## Request Body with Pydantic

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None

@app.post("/items/")
def create_item(item: Item):
    return {"item_name": item.name, "total": item.price + (item.tax or 0)}
```

Example request:
```json
{
    "name": "Laptop",
    "description": "A powerful laptop",
    "price": 1299.99,
    "tax": 129.99
}
```

## Path Parameters

```python
@app.get("/users/{user_id}")
def read_user(user_id: int):
    return {"user_id": user_id}

# With validation
from fastapi import Path

@app.get("/items/{item_id}")
def read_item(
    item_id: int = Path(..., title="The ID of the item", ge=1)
):
    return {"item_id": item_id}
```

## Query Parameters

```python
@app.get("/items/")
def read_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}

# Optional parameters
from typing import Optional

@app.get("/items/")
def read_items(q: Optional[str] = None):
    if q:
        return {"q": q}
    return {"q": "No query provided"}
```

## Request Validation

```python
from pydantic import BaseModel, Field, validator

class User(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str
    age: int = Field(..., ge=0, le=120)
    
    @validator('email')
    def email_must_contain_at(cls, v):
        if '@' not in v:
            raise ValueError('must contain @')
        return v

@app.post("/users/")
def create_user(user: User):
    return user
```

## Response Models

```python
class UserIn(BaseModel):
    username: str
    password: str
    email: str

class UserOut(BaseModel):
    username: str
    email: str

@app.post("/users/", response_model=UserOut)
def create_user(user: UserIn):
    # Password won't be included in response
    return user
```

## Dependency Injection

```python
from fastapi import Depends

def common_parameters(q: str = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/items/")
def read_items(commons: dict = Depends(common_parameters)):
    return commons

@app.get("/users/")
def read_users(commons: dict = Depends(common_parameters)):
    return commons
```

## Authentication Example

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if token != "secret-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return token

@app.get("/protected")
def protected_route(token: str = Depends(verify_token)):
    return {"message": "Access granted", "token": token}
```

## CORS (Cross-Origin Resource Sharing)

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Error Handling

```python
from fastapi import HTTPException

@app.get("/items/{item_id}")
def read_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return items_db[item_id]

# Custom exception handler
from fastapi.responses import JSONResponse

class CustomException(Exception):
    def __init__(self, name: str):
        self.name = name

@app.exception_handler(CustomException)
async def custom_exception_handler(request, exc: CustomException):
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.name} did something wrong."}
    )
```

## Background Tasks

```python
from fastapi import BackgroundTasks

def send_email(email: str, message: str):
    print(f"Sending email to {email}: {message}")

@app.post("/send-notification/")
def send_notification(email: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(send_email, email, "Notification sent!")
    return {"message": "Notification will be sent in background"}
```

## Database Integration (SQLAlchemy)

```python
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/users/")
def read_users(db: Session = Depends(get_db)):
    return db.query(User).all()
```

## File Uploads

```python
from fastapi import File, UploadFile

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(contents)
    }
```

## Production Deployment

### Using Uvicorn
```bash
# Development
uvicorn main:app --reload

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Gunicorn
```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Testing

```python
from fastapi.testclient import TestClient

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}

def test_create_item():
    response = client.post(
        "/items/",
        json={"name": "Test", "price": 10.5}
    )
    assert response.status_code == 200
    assert response.json()["item_name"] == "Test"
```

## Best Practices

1. **Use Pydantic models** for request/response validation
2. **Leverage dependency injection** for reusable logic
3. **Define response models** to control API contracts
4. **Use async/await** for I/O-bound operations
5. **Implement proper error handling** with HTTPException
6. **Add CORS middleware** for web applications
7. **Use environment variables** for configuration
8. **Write tests** with TestClient
9. **Document your API** with clear descriptions
10. **Use background tasks** for non-blocking operations

## Resources

- **Official Docs**: https://fastapi.tiangolo.com/
- **GitHub**: https://github.com/tiangolo/fastapi
- **Tutorial**: https://fastapi.tiangolo.com/tutorial/
- **Awesome FastAPI**: https://github.com/mjhea0/awesome-fastapi

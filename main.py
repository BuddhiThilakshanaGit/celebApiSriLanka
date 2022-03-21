from fastapi import FastAPI 
from model import predictCeleb,getCelebrities
from fastapi.middleware.cors import CORSMiddleware
from pydantic import   BaseModel



class Item(BaseModel):
    img: str
app = FastAPI()

origins = [
    "http://127.0.0.1:10807",
    "https://celebstatic.pages.dev",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/getCelebrities")
async def root():
    return getCelebrities()

@app.post("/predict")
async def root(item: Item):
    return predictCeleb(item.img)

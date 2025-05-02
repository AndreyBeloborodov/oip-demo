import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from txtai.embeddings import Embeddings

app = FastAPI()
templates = Jinja2Templates(directory="templates")

index = Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})
documents = [
    "Солнце светит ярко",
    "Сегодня хорошая погода",
    "Искусственный интеллект развивается быстро",
    "Машинное обучение используется в медицине",
    "Python — популярный язык программирования",
    "FastAPI — быстрый веб-фреймворк",
    "Векторный поиск улучшает результаты",
    "Docker упрощает деплой приложений",
    "Jinja2 используется для шаблонов",
    "txtai — легковесная альтернатива sentence-transformers"
]
index.index(documents)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search", response_class=HTMLResponse)
def search(request: Request, query: str = Form(...)):
    results = index.search(query, 10)
    matches = [(documents[r[0]], r[1]) for r in results]
    return templates.TemplateResponse("index.html", {"request": request, "results": matches, "query": query})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

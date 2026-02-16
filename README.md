# JusticeGraph API ⚖️

A minimal, production-ready ML API for predicting court backlogs and case duration.

## 🚀 How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Server
```bash
uvicorn justice_graph.api.main:app --reload
```
The API will be live at `http://127.0.0.1:8000`.

## 📡 Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict/backlog` | POST | Predicts risk level based on court stats. |
| `/predict/duration` | POST | Estimates case duration based on case details. |
| `/predict/district-backlog` | POST | Returns estimated duration for a specific district (Karnataka). |

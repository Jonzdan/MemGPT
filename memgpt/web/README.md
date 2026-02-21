# MemGPT Memory Log Dashboard

## Structure

```
memgpt-dashboard/
├── backend/
│   └── main.py          # FastAPI backend
├── frontend/
│   ├── index.html       # Main HTML
│   └── static/
│       ├── style.css    # Styles
│       └── app.js       # Frontend logic
└── requirements.txt
```

## Setup

```bash
pip install fastapi uvicorn
```

## Run

From inside your MemGPT project root:

```bash
cd C:\Users\jonzd\.vscode\HTML\cs6501-sais\MemGPT
uvicorn memgpt.dashboard:app --reload --port 8000
```

Then open http://localhost:8000

## Features

- **Overview** — Stats, operation breakdown by type
- **Sessions** — All sessions with op counts, label with attack scenario
- **Timeline** — Ordered event timeline for a session with context window % indicators  
- **Core Writes** — Filter core memory writes by time window
- **Search** — Full text search across log content
- **Diff Modal** — Side-by-side previous vs new value for any write operation
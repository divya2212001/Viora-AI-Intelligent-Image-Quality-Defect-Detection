# Viora AI — Intelligent Image Quality & Defect Detection

## 1. Overview

Viora AI analyzes an uploaded image with a hybrid ResNet18 and OpenCV-feature model. It returns a 0–100 quality score, a predicted qMOS score on a 0–5 scale, a quality label, eight image statistics, a recommendation, and scores for six defect categories: blur, underexposure, overexposure, noise, corruption, and defect. The application also creates a Grad-CAM visualization and keeps session-scoped analysis history in MongoDB.

## 2. Key Features

- Image upload (JPEG, PNG, WebP, and BMP API support; the UI offers JPEG, PNG, and WebP)
- Quality prediction: quality score, qMOS, and label
- Six defect scores: blur, underexposure, overexposure, noise, corruption, and defect
- OpenCV statistics: brightness, contrast, sharpness, noise level, entropy, saturation, dark-pixel ratio, and bright-pixel ratio
- Recommendation, original-image rendering, and Grad-CAM explainability
- MongoDB-backed, session-scoped analysis history, reopening, and deletion
- FastAPI REST API, React/Vite frontend, and a model-information endpoint

## 3. Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React 19, React Router, Vite |
| Backend | Python, FastAPI, Uvicorn |
| Deep learning | PyTorch, Torchvision ResNet18 |
| Image processing | OpenCV, NumPy |
| Database | MongoDB / PyMongo |
| Explainability | Grad-CAM implemented with PyTorch hooks |
| Containers | Docker, Docker Compose |

## 4. Architecture

```text
User
  ↓
React frontend (upload, results, history)
  ↓ HTTP multipart/form-data + session_id
FastAPI
  ↓
Prediction service
  ├─ ResNet18 image branch
  ├─ OpenCV feature branch
  ├─ feature fusion
  ├─ qMOS/quality output
  ├─ six defect outputs
  └─ Grad-CAM generation
  ↓
MongoDB prediction history + local /uploads static files
```

## 5. Project Structure

```text
backend/
  app/                 FastAPI routes, services, MongoDB access
  ml/                  model, data pipeline, artifacts, and reports
  requirements.txt
frontend/
  src/                 React pages, components, API client
  package.json
docker-compose.yml
DOCUMENTATION.md       detailed technical documentation
```

## 6. Quick Start

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit MONGODB_URI and other values in .env as needed.
PYTHONPATH=. uvicorn app.main:app --reload
```

The API is served at `http://127.0.0.1:8000`; Swagger UI is at `http://127.0.0.1:8000/docs`.

### Frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

The Vite development server defaults to `http://localhost:5173`.

## 7. Environment Variables

Backend variables (`backend/.env`):

| Variable | Default / example | Purpose |
|---|---|---|
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `DATABASE_NAME` | `image_quality_db` | Database name |
| `MAX_FILE_SIZE_MB` | `10` | Maximum uploaded-file size |
| `UPLOAD_DIR` | `uploads` | Local original/Grad-CAM storage |
| `APP_NAME`, `APP_VERSION` | project defaults | API metadata |
| `MODEL_NAME`, `MODEL_VERSION` | project defaults | service metadata |
| `CORS_ORIGINS` | comma-separated URLs | allowed browser origins |

For a hosted MongoDB instance, use `MONGODB_URI=mongodb+srv://<username>:<password>@<cluster>/<database>`; do not commit real credentials. The frontend accepts `VITE_API_BASE_URL=http://localhost:8000`.

## 8. Database

MongoDB stores successful prediction documents in the `predictions` collection. Start a local server with your MongoDB installation, or use the `mongodb` service in the Compose file. History requests require the browser-generated `session_id`; details are in [Detailed Technical Documentation](DOCUMENTATION.md).

## 9. API Quick Reference

All `/api` history and prediction endpoints below require a non-empty `session_id` query parameter.

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/predict` | Analyze an image |
| `GET` | `/api/history` | List the current session’s analyses |
| `GET` | `/api/analyses/{prediction_id}` | Read one current-session analysis |
| `DELETE` | `/api/analyses/{prediction_id}` | Delete one current-session analysis and its files |
| `GET` | `/api/model-info` | Read active model metadata |
| `GET` | `/health` | Read model and database availability |

```bash
curl -X POST "http://127.0.0.1:8000/api/predict?session_id=<session-id>" \
  -H "accept: application/json" \
  -F "file=@sample.jpg"
```

## 10. Model Summary

The active checkpoint is a hybrid model: a pretrained ResNet18 produces an image embedding and an eight-value OpenCV branch produces a handcrafted-feature embedding. They are fused for a normalized qMOS regression output and six sigmoid defect outputs. The feature vectors are normalized with stored training-set mean and standard-deviation arrays.

## 11. Evaluation Summary

Quality metrics are qMOS-scale metrics from the generated reports on 1,511 held-out samples.

| Model | MAE | RMSE | PLCC | SRCC |
|---|---:|---:|---:|---:|
| CV baseline | 0.3958 | 0.5086 | 0.6894 | 0.6736 |
| CNN-only ResNet18 | 0.2993 | 0.3851 | 0.8367 | 0.8009 |
| Hybrid CNN + CV | 0.2883 | 0.3649 | 0.8566 | 0.8156 |

The separately generated six-label candidate evaluation uses deterministic synthetic proxy targets, not human defect annotations. Its F1@0.5 is 0.9676 (blur), 0.5460 (underexposure), 0.5385 (overexposure), 0.8458 (noise), 0.8440 (corruption), and 0.1143 (defect). See the detailed report interpretation in [DOCUMENTATION.md](DOCUMENTATION.md).

## 12. Sample Images

Generated report images are included in the repository:

![Best quality-prediction cases](backend/ml/reports/best_cases_contact_sheet.png)

![Worst quality-prediction cases](backend/ml/reports/worst_cases_contact_sheet.png)

## 13. Docker

Compose defines MongoDB (27017), the backend (8000), and the Vite frontend (5173):

```bash
docker compose build
docker compose up
docker compose down
```

**Current limitation:** the backend Dockerfile does not copy `backend/ml/artifacts/`, so the backend container cannot load its checkpoint as written. Use the local setup above, or amend the image to include the model artifacts before relying on Compose for inference.

## 14. Deployment

No complete cloud deployment configuration is present. The repository provides Dockerfiles, Compose configuration, configurable CORS origins, configurable API URL, and local static upload storage; it does not provide a deployment URL or provider-specific configuration.

## 15. Documentation

[Detailed Technical Documentation](DOCUMENTATION.md)

## 16. License / Author

No license or author metadata was found in the repository.

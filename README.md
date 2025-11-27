# Breast Cancer ML Service

A minimal production-style ML prediction service built with:

- **scikit-learn** for modeling
- **FastAPI** for serving
- **Docker** for containerization
- **YAML** config + small config layer
- Basic tests with **FastAPI TestClient**

The goal is not to optimize model performance, but to demonstrate
clean ML engineering practices: separation between training, prediction,
and serving, plus reproducible deployment.

---

## Project Structure

```text
ml_service/
├── src/
│   └── ml_service/
│       ├── config.py      # load config from YAML
│       ├── model.py       # train and save model
│       ├── predict.py     # prediction logic + pydantic models
│       └── server.py      # FastAPI app
├── configs/
│   └── config.yaml        # model path and training hyperparams
├── models/
│   └── model.pkl          # trained model (generated)
├── tests/
│   └── test_predict.py    # basic API tests
├── requirements.txt
├── Dockerfile
└── README.md

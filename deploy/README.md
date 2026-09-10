# Deploy (optional)

Infrastructure and MLOps assets for deploying G-RAG beyond local development.

| Path | Purpose |
|------|---------|
| `helm/` | Helm chart |
| `k8s/` | Raw Kubernetes manifests |
| `terraform/` | AWS free-tier style provisioning |
| `monitoring/` | Prometheus / Grafana values and dashboard |
| `Jenkinsfile*` | CI/CD pipelines |

**Local development does not require this folder.**

Use Docker Compose at the repository root:

```bash
docker compose up --build
```

Then pull a model into Ollama:

```bash
docker exec -it grag-ollama ollama run phi3:mini
```

Open the app at `http://localhost:8001`.

Paths inside these files may still reference older names (`rag-assistant`, Streamlit frontend). Update them before production use.

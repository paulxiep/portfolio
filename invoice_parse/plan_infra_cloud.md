# Infrastructure: Cloud IaC & CI/CD

## Purpose
Define everything needed to deploy the locally-working system to cloud. Covers infrastructure provisioning, container deployment, secrets, and CI/CD pipeline.

**Build after services work locally end-to-end.**

---

## Tech Stack (Cloud)

| Component | Local | Cloud (AWS) |
|-----------|-------|-------------|
| Database | SQLite | RDS Postgres |
| Blob storage | Local filesystem | S3 |
| Queue | Redis | SQS |
| Orchestration | Docker Compose | ECS Fargate |
| Container registry | Local images | ECR |
| Secrets | `.env` / config files | Secrets Manager / SSM Parameter Store |
| Monitoring | Logs to stdout | CloudWatch |
| IaC | N/A | Terraform |

---

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| AWS as primary cloud target | Broad service coverage, ECS Fargate for containers |
| Terraform for IaC | Declarative, well-documented, team-friendly |
| ECS Fargate over EKS | Simpler for this scale; no cluster management overhead |
| SQS over managed Redis | Fully managed, no server to maintain, native dead-letter queues |
| GitHub Actions for CI/CD | Integrated with repo, free tier sufficient |

---

## Cloud IaC (Terraform)

### Resources to provision
- **Networking**: VPC, subnets, security groups
- **Database**: RDS Postgres instance, parameter group
- **Storage**: S3 bucket with tenant-prefixed paths
- **Queue**: SQS queues (Queue A, Queue B) + dead-letter queues
- **Compute**: ECS cluster, task definitions for each service, Fargate launch type
- **Container registry**: ECR repositories per service
- **Load balancer**: ALB for Ingestion service (webhook endpoint)
- **Secrets**: Secrets Manager for API keys (Telegram bot token, Gemini API key)

### Terraform structure
```
infra/terraform/
├── main.tf
├── variables.tf
├── outputs.tf
├── modules/
│   ├── networking/
│   ├── database/
│   ├── storage/
│   ├── queue/
│   ├── ecs/
│   └── secrets/
└── environments/
    ├── staging.tfvars
    └── production.tfvars
```

---

## Container Registry (ECR)

### Repositories
- `invoice-ingestion` (Rust)
- `invoice-processing` (Python)
- `invoice-output` (Rust)
- `invoice-dashboard` (Python/Streamlit)

### Image tagging strategy
- `latest` for current main branch
- `sha-<commit>` for traceability
- `v<semver>` for releases

---

## CI/CD Pipeline (GitHub Actions)

### Triggers
- Push to `main` → build + deploy to staging
- Tag `v*` → deploy to production
- PR → build + test only

### Pipeline stages
1. **Lint & test** — per-service unit tests, schema validation
2. **Build** — Docker images for all services
3. **Push** — to ECR
4. **Deploy** — `terraform apply` + ECS service update

### Per-service Dockerfiles
- Rust services: multi-stage build (build in `rust:latest`, run in `debian-slim`)
- Python services: `python:3.12-slim`, pip install from `requirements.txt`

---

## Secrets Management

| Secret | Storage | Injected via |
|--------|---------|--------------|
| Telegram bot token | Secrets Manager | ECS task definition env var |
| Gemini API key | Secrets Manager | ECS task definition env var |
| Database connection string | SSM Parameter Store | ECS task definition env var |
| S3 bucket name | Terraform output | ECS task definition env var |
| SQS queue URLs | Terraform output | ECS task definition env var |

---

## Monitoring & Logging

- **Logs**: All services log to stdout → CloudWatch Logs via ECS log driver
- **Metrics**: CloudWatch custom metrics for queue depth, processing latency, error rates
- **Alerts**: CloudWatch alarms for:
  - Queue depth > threshold (processing backlog)
  - Error rate > threshold
  - ECS task failures
  - RDS connection/storage thresholds

---

## POC Scope
- [ ] Terraform modules for core resources (VPC, RDS, S3, SQS, ECS)
- [ ] ECR repositories + Dockerfiles per service
- [ ] GitHub Actions workflow for build + push
- [ ] Basic CloudWatch log groups
- [ ] Manual deploy initially, automated deploy as stretch goal

## Production Considerations
- Multi-AZ RDS for availability
- S3 lifecycle policies for old job artifacts
- SQS dead-letter queues with alerting
- Auto-scaling ECS tasks based on SQS queue depth
- Cost monitoring and budgets
- Staging environment for pre-production validation

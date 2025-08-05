# Hybrid AI Architecture - Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Hybrid AI Architecture across different environments, from local development to enterprise production. Choose the deployment option that best fits your needs and infrastructure requirements.

## 🚀 Deployment Options

### 1. **Local Development Deployment** (Quickest Start)
### 2. **Cloud Deployment** (Recommended for Production)
### 3. **On-Premises Enterprise Deployment** (Maximum Control)
### 4. **Hybrid Cloud Deployment** (Best of Both Worlds)

---

## 🏠 Option 1: Local Development Deployment

### Prerequisites
- **Docker Desktop** with Kubernetes enabled
- **Python 3.9+** with pip
- **Git** for version control
- **8GB+ RAM** and **4+ CPU cores**
- **50GB+ free disk space**

### Quick Start (5 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/your-org/hybrid-ai-architecture.git
cd hybrid-ai-architecture

# 2. Set up environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Edit .env with your API keys:
# OPENAI_API_KEY=your_openai_key
# ANTHROPIC_API_KEY=your_anthropic_key
# GOOGLE_API_KEY=your_google_key

# 4. Start local services
docker-compose up -d

# 5. Initialize database
python scripts/init_database.py

# 6. Start the application
python main.py
```

### Access Points
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Admin Dashboard**: http://localhost:8000/admin
- **Monitoring**: http://localhost:3000 (Grafana)

---

## ☁️ Option 2: Cloud Deployment (Recommended)

### 2A. AWS Deployment

#### Prerequisites
- **AWS Account** with appropriate permissions
- **AWS CLI** configured
- **kubectl** installed
- **Helm 3.x** installed

#### Infrastructure Setup

```bash
# 1. Create EKS cluster
eksctl create cluster \
  --name hybrid-ai-cluster \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type m5.xlarge \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10 \
  --managed

# 2. Configure kubectl
aws eks update-kubeconfig --region us-west-2 --name hybrid-ai-cluster

# 3. Install GPU support (for model inference)
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.12.2/nvidia-device-plugin.yml

# 4. Install ingress controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/aws/deploy.yaml
```

#### Application Deployment

```bash
# 1. Create namespace
kubectl create namespace hybrid-ai

# 2. Create secrets
kubectl create secret generic api-keys \
  --from-literal=openai-key=$OPENAI_API_KEY \
  --from-literal=anthropic-key=$ANTHROPIC_API_KEY \
  --from-literal=google-key=$GOOGLE_API_KEY \
  -n hybrid-ai

# 3. Deploy using Helm
helm repo add hybrid-ai https://charts.hybrid-ai.com
helm install hybrid-ai hybrid-ai/hybrid-ai-platform \
  --namespace hybrid-ai \
  --set global.environment=production \
  --set ingress.enabled=true \
  --set ingress.hostname=your-domain.com

# 4. Wait for deployment
kubectl wait --for=condition=available --timeout=600s deployment --all -n hybrid-ai
```

#### AWS-Specific Services

```yaml
# aws-resources.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: aws-config
  namespace: hybrid-ai
data:
  # RDS Database
  database_host: "hybrid-ai-db.cluster-xyz.us-west-2.rds.amazonaws.com"
  database_port: "5432"
  
  # ElastiCache Redis
  redis_host: "hybrid-ai-cache.xyz.cache.amazonaws.com"
  redis_port: "6379"
  
  # S3 Storage
  s3_bucket: "hybrid-ai-models-bucket"
  s3_region: "us-west-2"
  
  # CloudWatch Monitoring
  cloudwatch_region: "us-west-2"
```

### 2B. Google Cloud Platform (GCP) Deployment

```bash
# 1. Create GKE cluster
gcloud container clusters create hybrid-ai-cluster \
  --zone us-central1-a \
  --machine-type n1-standard-4 \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10 \
  --enable-autorepair \
  --enable-autoupgrade

# 2. Get credentials
gcloud container clusters get-credentials hybrid-ai-cluster --zone us-central1-a

# 3. Deploy application
kubectl apply -f k8s/gcp/
```

### 2C. Microsoft Azure Deployment

```bash
# 1. Create AKS cluster
az aks create \
  --resource-group hybrid-ai-rg \
  --name hybrid-ai-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# 2. Get credentials
az aks get-credentials --resource-group hybrid-ai-rg --name hybrid-ai-cluster

# 3. Deploy application
kubectl apply -f k8s/azure/
```

---

## 🏢 Option 3: On-Premises Enterprise Deployment

### Prerequisites
- **Kubernetes cluster** (v1.24+)
- **GPU nodes** (NVIDIA A100/V100 recommended)
- **High-performance storage** (NVMe SSD)
- **Load balancer** (MetalLB or hardware)
- **Container registry** (Harbor or similar)

### Infrastructure Requirements

```yaml
# Minimum Hardware Requirements
nodes:
  master_nodes: 3
  worker_nodes: 5
  gpu_nodes: 2

compute:
  cpu_cores: 64 total
  memory: 256GB total
  gpu_memory: 80GB total

storage:
  persistent_storage: 2TB SSD
  backup_storage: 5TB
  model_storage: 1TB NVMe

network:
  bandwidth: 10Gbps
  load_balancer: Hardware or MetalLB
```

### Deployment Steps

```bash
# 1. Prepare container images
docker build -t your-registry.com/hybrid-ai/orchestrator:latest .
docker push your-registry.com/hybrid-ai/orchestrator:latest

# 2. Create namespace and RBAC
kubectl apply -f k8s/enterprise/namespace.yaml
kubectl apply -f k8s/enterprise/rbac.yaml

# 3. Deploy storage classes
kubectl apply -f k8s/enterprise/storage-classes.yaml

# 4. Deploy databases
kubectl apply -f k8s/enterprise/postgresql-cluster.yaml
kubectl apply -f k8s/enterprise/redis-cluster.yaml

# 5. Deploy monitoring stack
kubectl apply -f k8s/enterprise/monitoring/

# 6. Deploy application
kubectl apply -f k8s/enterprise/applications/

# 7. Configure ingress
kubectl apply -f k8s/enterprise/ingress.yaml
```

### Enterprise Security Configuration

```yaml
# security-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: security-config
  namespace: hybrid-ai
data:
  # TLS Configuration
  tls_enabled: "true"
  tls_cert_path: "/etc/ssl/certs/hybrid-ai.crt"
  tls_key_path: "/etc/ssl/private/hybrid-ai.key"
  
  # Authentication
  auth_provider: "ldap"
  ldap_server: "ldap://your-ldap-server.com"
  
  # Network Policies
  network_policies_enabled: "true"
  
  # Audit Logging
  audit_logging_enabled: "true"
  audit_log_path: "/var/log/audit/hybrid-ai.log"
```

---

## 🌐 Option 4: Hybrid Cloud Deployment

### Architecture Overview
- **Control Plane**: On-premises for security and compliance
- **Compute Nodes**: Cloud for scalability and cost optimization
- **Data Storage**: Hybrid with sensitive data on-premises

### Setup Steps

```bash
# 1. Set up on-premises control plane
kubeadm init --control-plane-endpoint=your-control-plane.com

# 2. Join cloud nodes
# On each cloud node:
kubeadm join your-control-plane.com:6443 --token <token> --discovery-token-ca-cert-hash <hash>

# 3. Configure hybrid networking
kubectl apply -f k8s/hybrid/network-config.yaml

# 4. Deploy with node affinity
kubectl apply -f k8s/hybrid/hybrid-deployment.yaml
```

---

## 🔧 Configuration Management

### Environment-Specific Configurations

#### Development Environment
```yaml
# config/development.yaml
environment: development
debug: true
log_level: DEBUG

database:
  host: localhost
  port: 5432
  name: hybrid_ai_dev

models:
  cache_enabled: false
  mock_responses: true

monitoring:
  metrics_enabled: false
```

#### Staging Environment
```yaml
# config/staging.yaml
environment: staging
debug: false
log_level: INFO

database:
  host: staging-db.internal
  port: 5432
  name: hybrid_ai_staging

models:
  cache_enabled: true
  mock_responses: false

monitoring:
  metrics_enabled: true
```

#### Production Environment
```yaml
# config/production.yaml
environment: production
debug: false
log_level: WARNING

database:
  host: prod-db-cluster.internal
  port: 5432
  name: hybrid_ai_prod

models:
  cache_enabled: true
  cache_ttl: 3600
  mock_responses: false

monitoring:
  metrics_enabled: true
  alerting_enabled: true

security:
  tls_enabled: true
  auth_required: true
  rate_limiting: true
```

---

## 📊 Monitoring and Observability Setup

### Prometheus Configuration
```yaml
# monitoring/prometheus-config.yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'hybrid-ai-orchestrator'
    static_configs:
      - targets: ['orchestrator-service:8080']
  
  - job_name: 'hybrid-ai-models'
    static_configs:
      - targets: ['model-service:8080']
  
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
```

### Grafana Dashboards
```bash
# Import pre-built dashboards
kubectl create configmap grafana-dashboards \
  --from-file=monitoring/dashboards/ \
  -n hybrid-ai

# Configure data sources
kubectl apply -f monitoring/grafana-datasources.yaml
```

---

## 🔐 Security Hardening

### Network Security
```yaml
# security/network-policies.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: hybrid-ai-network-policy
  namespace: hybrid-ai
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: hybrid-ai
    ports:
    - protocol: TCP
      port: 8080
```

### Pod Security Standards
```yaml
# security/pod-security.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: hybrid-ai
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

---

## 🚀 Deployment Automation

### CI/CD Pipeline (GitHub Actions)
```yaml
# .github/workflows/deploy.yml
name: Deploy Hybrid AI Architecture

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-west-2
    
    - name: Deploy to EKS
      run: |
        aws eks update-kubeconfig --name hybrid-ai-cluster
        helm upgrade --install hybrid-ai ./helm/hybrid-ai \
          --namespace hybrid-ai \
          --set image.tag=${{ github.sha }}
```

### Terraform Infrastructure as Code
```hcl
# terraform/main.tf
provider "aws" {
  region = "us-west-2"
}

module "eks" {
  source = "terraform-aws-modules/eks/aws"
  
  cluster_name    = "hybrid-ai-cluster"
  cluster_version = "1.27"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  node_groups = {
    standard = {
      desired_capacity = 3
      max_capacity     = 10
      min_capacity     = 1
      instance_types   = ["m5.xlarge"]
    }
    
    gpu = {
      desired_capacity = 1
      max_capacity     = 3
      min_capacity     = 0
      instance_types   = ["p3.2xlarge"]
    }
  }
}
```

---

## 📋 Pre-Deployment Checklist

### Infrastructure Readiness
- [ ] Kubernetes cluster running (v1.24+)
- [ ] GPU nodes available for model inference
- [ ] Persistent storage configured
- [ ] Load balancer configured
- [ ] DNS records configured
- [ ] SSL certificates obtained

### Security Requirements
- [ ] Network policies configured
- [ ] RBAC permissions set up
- [ ] Secrets management configured
- [ ] Container image scanning enabled
- [ ] Audit logging configured

### Application Configuration
- [ ] API keys configured
- [ ] Database connections tested
- [ ] Model artifacts uploaded
- [ ] Configuration files validated
- [ ] Health checks configured

### Monitoring Setup
- [ ] Prometheus deployed
- [ ] Grafana dashboards imported
- [ ] Alerting rules configured
- [ ] Log aggregation set up
- [ ] Backup procedures tested

---

## 🆘 Troubleshooting Common Issues

### Pod Startup Issues
```bash
# Check pod status
kubectl get pods -n hybrid-ai

# View pod logs
kubectl logs -f deployment/ai-orchestrator -n hybrid-ai

# Describe pod for events
kubectl describe pod <pod-name> -n hybrid-ai
```

### Database Connection Issues
```bash
# Test database connectivity
kubectl exec -it postgres-0 -n hybrid-ai -- psql -U admin -d hybrid_ai -c "SELECT 1;"

# Check database service
kubectl get svc postgres-service -n hybrid-ai
```

### Model Loading Issues
```bash
# Check GPU availability
kubectl exec -it model-pod -n hybrid-ai -- nvidia-smi

# Verify model artifacts
kubectl exec -it model-pod -n hybrid-ai -- ls -la /models/
```

---

## 📞 Support and Resources

### Documentation
- **Technical Docs**: [docs/technical/system_architecture.md](docs/technical/system_architecture.md)
- **API Reference**: [docs/api/api_documentation.md](docs/api/api_documentation.md)
- **Troubleshooting**: [docs/troubleshooting/runbooks.md](docs/troubleshooting/runbooks.md)

### Community Support
- **GitHub Issues**: https://github.com/your-org/hybrid-ai-architecture/issues
- **Slack Community**: https://hybrid-ai-community.slack.com
- **Documentation**: https://docs.hybrid-ai.com

### Enterprise Support
- **Email**: enterprise-support@hybrid-ai.com
- **Phone**: 1-800-HYBRID-AI
- **Professional Services**: Available for deployment assistance

---

## 🎯 Recommended Deployment Path

### For Development/Testing
1. **Start with Local Development** - Quick setup for testing
2. **Move to Cloud Staging** - Test in cloud environment
3. **Deploy to Production** - Full production deployment

### For Enterprise
1. **Proof of Concept** - Local or small cloud deployment
2. **Pilot Deployment** - Limited production deployment
3. **Full Production** - Enterprise-scale deployment with all features

### Timeline Estimates
- **Local Development**: 30 minutes
- **Cloud Deployment**: 2-4 hours
- **Enterprise Deployment**: 1-2 days
- **Full Production with Training**: 1-2 weeks

Choose the deployment option that best fits your requirements, infrastructure, and timeline. The system is designed to be flexible and can be deployed across various environments while maintaining full functionality and performance.
# System Administrator Training Guide

## Overview

This comprehensive training guide provides system administrators with the knowledge and skills needed to effectively operate, maintain, and troubleshoot the Hybrid AI Architecture platform. The guide covers installation, configuration, monitoring, maintenance, and advanced troubleshooting procedures.

## Prerequisites

### Required Knowledge
- **Kubernetes**: Container orchestration and cluster management
- **Docker**: Container technology and image management
- **Linux/Unix**: Command line operations and system administration
- **Networking**: TCP/IP, DNS, load balancing, and security
- **Databases**: PostgreSQL administration and optimization
- **Monitoring**: Prometheus, Grafana, and observability tools

### Required Access
- **Kubernetes Cluster**: Admin access to production and staging clusters
- **Cloud Platform**: AWS/GCP/Azure administrative access
- **Monitoring Systems**: Access to Prometheus, Grafana, and alerting systems
- **CI/CD Pipeline**: Access to deployment and build systems
- **Documentation**: Access to internal documentation and runbooks

## Module 1: System Architecture Overview

### Core Components Understanding

#### 1. AI Orchestrator
**Purpose**: Central coordination hub for all AI operations
**Key Responsibilities**:
- Request routing and load balancing
- Model selection and orchestration
- Response synthesis and validation
- Performance monitoring and optimization

**Administrative Tasks**:
```bash
# Check orchestrator health
kubectl get pods -l app=ai-orchestrator -n hybrid-ai

# View orchestrator logs
kubectl logs -f deployment/ai-orchestrator -n hybrid-ai

# Scale orchestrator instances
kubectl scale deployment ai-orchestrator --replicas=5 -n hybrid-ai
```

#### 2. Model Infrastructure
**Components**:
- Foundation model integrations (GPT-4, Gemini, Claude)
- Specialized models (Earnings, Sentiment, Risk, Thematic)
- Model registry and versioning system
- GPU resource management

**Administrative Tasks**:
```bash
# Check model pod status
kubectl get pods -l component=model -n hybrid-ai

# Monitor GPU utilization
kubectl exec -it earnings-analyzer-pod -- nvidia-smi

# Update model version
kubectl set image deployment/earnings-analyzer \
  earnings-analyzer=models/earnings-analyzer:v2.1.0
```

#### 3. Data Infrastructure
**Components**:
- PostgreSQL for structured data
- InfluxDB for time-series metrics
- Redis for caching and sessions
- Object storage for model artifacts

**Administrative Tasks**:
```bash
# Check database health
kubectl exec -it postgres-primary-0 -- \
  psql -U admin -d hybrid_ai -c "SELECT version();"

# Monitor Redis performance
kubectl exec -it redis-0 -- redis-cli info stats

# Check storage utilization
kubectl exec -it postgres-primary-0 -- df -h /var/lib/postgresql/data
```

### System Architecture Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer                            │
├─────────────────────────────────────────────────────────────┤
│                    API Gateway                              │
├─────────────────────────────────────────────────────────────┤
│                  AI Orchestrator                            │
├─────────────────────────────────────────────────────────────┤
│  Foundation Models    │    Specialized Models               │
│  ┌─────────────────┐  │  ┌─────────────────────────────────┐ │
│  │ GPT-4  Gemini   │  │  │ Earnings  Sentiment  Risk       │ │
│  │ Claude          │  │  │ Thematic  Trend                 │ │
│  └─────────────────┘  │  └─────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Human-in-Loop    │  Guardrails    │  Monitoring           │
├─────────────────────────────────────────────────────────────┤
│  PostgreSQL       │  InfluxDB      │  Redis                │
└─────────────────────────────────────────────────────────────┘
```

## Module 2: Installation and Configuration

### Initial System Setup

#### 1. Kubernetes Cluster Preparation
```bash
# Create namespace
kubectl create namespace hybrid-ai

# Set up RBAC
kubectl apply -f k8s/rbac/

# Install cert-manager for TLS
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Install ingress controller
kubectl apply -f k8s/ingress/nginx-ingress.yaml
```

#### 2. Storage Configuration
```bash
# Create storage classes
kubectl apply -f k8s/storage/storage-classes.yaml

# Set up persistent volumes
kubectl apply -f k8s/storage/persistent-volumes.yaml

# Configure backup storage
kubectl apply -f k8s/storage/backup-storage.yaml
```

#### 3. Database Setup
```bash
# Deploy PostgreSQL cluster
kubectl apply -f k8s/databases/postgresql-cluster.yaml

# Wait for cluster to be ready
kubectl wait --for=condition=ready pod -l app=postgresql -n hybrid-ai --timeout=300s

# Initialize database schema
kubectl exec -it postgres-primary-0 -n hybrid-ai -- \
  psql -U admin -d hybrid_ai -f /scripts/init-schema.sql

# Create database users
kubectl exec -it postgres-primary-0 -n hybrid-ai -- \
  psql -U admin -d hybrid_ai -f /scripts/create-users.sql
```

#### 4. Redis Configuration
```bash
# Deploy Redis cluster
kubectl apply -f k8s/databases/redis-cluster.yaml

# Configure Redis for high availability
kubectl apply -f k8s/databases/redis-sentinel.yaml

# Test Redis connectivity
kubectl exec -it redis-0 -n hybrid-ai -- redis-cli ping
```

### Application Deployment

#### 1. Core Services Deployment
```bash
# Deploy AI Orchestrator
kubectl apply -f k8s/services/ai-orchestrator.yaml

# Deploy API Gateway
kubectl apply -f k8s/services/api-gateway.yaml

# Deploy Guardrail Engine
kubectl apply -f k8s/services/guardrail-engine.yaml

# Deploy Human-in-the-Loop system
kubectl apply -f k8s/services/human-review-system.yaml
```

#### 2. Model Services Deployment
```bash
# Deploy foundation model integrations
kubectl apply -f k8s/models/foundation-models.yaml

# Deploy specialized models
kubectl apply -f k8s/models/specialized-models.yaml

# Configure model registry
kubectl apply -f k8s/models/model-registry.yaml
```

#### 3. Monitoring Stack Deployment
```bash
# Deploy Prometheus
kubectl apply -f k8s/monitoring/prometheus.yaml

# Deploy Grafana
kubectl apply -f k8s/monitoring/grafana.yaml

# Deploy AlertManager
kubectl apply -f k8s/monitoring/alertmanager.yaml

# Import dashboards
kubectl create configmap grafana-dashboards \
  --from-file=monitoring/dashboards/ -n hybrid-ai
```

### Configuration Management

#### 1. Environment Configuration
```yaml
# config/production.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: hybrid-ai
data:
  environment: "production"
  log_level: "INFO"
  database_url: "postgresql://admin:password@postgres-primary:5432/hybrid_ai"
  redis_url: "redis://redis-sentinel:26379"
  model_registry_url: "http://model-registry:8080"
  
  # AI Model Configuration
  openai_api_key: "sk-..."
  anthropic_api_key: "sk-ant-..."
  google_api_key: "AIza..."
  
  # Performance Settings
  max_concurrent_requests: "1000"
  request_timeout: "30s"
  model_cache_ttl: "3600"
  
  # Security Settings
  jwt_secret: "your-jwt-secret"
  encryption_key: "your-encryption-key"
  cors_origins: "https://app.hybrid-ai.com"
```

#### 2. Model Configuration
```yaml
# config/models.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
  namespace: hybrid-ai
data:
  models.yaml: |
    foundation_models:
      gpt4:
        endpoint: "https://api.openai.com/v1"
        model: "gpt-4-turbo-preview"
        max_tokens: 4096
        temperature: 0.1
        
      gemini:
        endpoint: "https://generativelanguage.googleapis.com/v1"
        model: "gemini-pro"
        max_tokens: 2048
        temperature: 0.1
        
    specialized_models:
      earnings_analyzer:
        image: "models/earnings-analyzer:v2.1.0"
        replicas: 3
        resources:
          requests:
            memory: "4Gi"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            nvidia.com/gpu: 1
            
      sentiment_analyzer:
        image: "models/sentiment-analyzer:v1.8.0"
        replicas: 2
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

## Module 3: Monitoring and Observability

### Metrics Collection and Analysis

#### 1. Key Performance Indicators (KPIs)
```bash
# Response time metrics
kubectl exec -it prometheus-0 -n monitoring -- \
  promtool query instant 'histogram_quantile(0.95, http_request_duration_seconds_bucket)'

# Error rate metrics
kubectl exec -it prometheus-0 -n monitoring -- \
  promtool query instant 'rate(http_requests_total{status=~"5.."}[5m])'

# Model accuracy metrics
kubectl exec -it prometheus-0 -n monitoring -- \
  promtool query instant 'avg(model_accuracy_score) by (model_name)'
```

#### 2. System Health Monitoring
```bash
# Check cluster resource utilization
kubectl top nodes
kubectl top pods -n hybrid-ai

# Monitor database performance
kubectl exec -it postgres-primary-0 -n hybrid-ai -- \
  psql -U admin -d hybrid_ai -c "
    SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del 
    FROM pg_stat_user_tables 
    ORDER BY n_tup_ins DESC LIMIT 10;"

# Check Redis memory usage
kubectl exec -it redis-0 -n hybrid-ai -- \
  redis-cli info memory | grep used_memory_human
```

#### 3. Application Metrics
```bash
# Model inference metrics
kubectl exec -it prometheus-0 -n monitoring -- \
  promtool query instant 'rate(model_inference_total[5m])'

# Queue depth monitoring
kubectl exec -it prometheus-0 -n monitoring -- \
  promtool query instant 'review_queue_depth'

# Cache hit rate
kubectl exec -it prometheus-0 -n monitoring -- \
  promtool query instant 'cache_hit_rate'
```

### Dashboard Configuration

#### 1. Executive Dashboard Setup
```json
{
  "dashboard": {
    "title": "Hybrid AI - Executive Dashboard",
    "panels": [
      {
        "title": "System Availability",
        "type": "stat",
        "targets": [
          {
            "expr": "avg(up{job=\"hybrid-ai\"})",
            "legendFormat": "Uptime %"
          }
        ]
      },
      {
        "title": "Request Volume",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Model Accuracy",
        "type": "graph",
        "targets": [
          {
            "expr": "avg(model_accuracy_score) by (model_name)",
            "legendFormat": "{{model_name}}"
          }
        ]
      }
    ]
  }
}
```

#### 2. Technical Dashboard Setup
```json
{
  "dashboard": {
    "title": "Hybrid AI - Technical Dashboard",
    "panels": [
      {
        "title": "Response Time Distribution",
        "type": "heatmap",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, http_request_duration_seconds_bucket)",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate by Service",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) by (service)",
            "legendFormat": "{{service}}"
          }
        ]
      },
      {
        "title": "Resource Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(container_cpu_usage_seconds_total[5m]) by (pod)",
            "legendFormat": "{{pod}}"
          }
        ]
      }
    ]
  }
}
```

### Alerting Configuration

#### 1. Critical Alerts
```yaml
# alerts/critical.yaml
groups:
- name: critical
  rules:
  - alert: SystemDown
    expr: up{job="hybrid-ai"} == 0
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Hybrid AI system is down"
      description: "System has been down for more than 5 minutes"
      
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
    for: 10m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} which is above 5%"
      
  - alert: DatabaseConnectionFailure
    expr: postgresql_up == 0
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Database connection failure"
      description: "Cannot connect to PostgreSQL database"
```

#### 2. Warning Alerts
```yaml
# alerts/warning.yaml
groups:
- name: warning
  rules:
  - alert: HighResponseTime
    expr: histogram_quantile(0.95, http_request_duration_seconds_bucket) > 1.0
    for: 15m
    labels:
      severity: warning
    annotations:
      summary: "High response time detected"
      description: "95th percentile response time is {{ $value }}s"
      
  - alert: ModelAccuracyDrop
    expr: model_accuracy_score < 0.85
    for: 30m
    labels:
      severity: warning
    annotations:
      summary: "Model accuracy drop detected"
      description: "Model {{ $labels.model_name }} accuracy is {{ $value }}"
      
  - alert: HighMemoryUsage
    expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage"
      description: "Container {{ $labels.container }} memory usage is {{ $value }}%"
```

## Module 4: Maintenance Procedures

### Routine Maintenance Tasks

#### 1. Daily Maintenance Checklist
```bash
#!/bin/bash
# daily-maintenance.sh

echo "=== Daily Maintenance Checklist ==="

# 1. Check system health
echo "1. Checking system health..."
kubectl get pods -n hybrid-ai | grep -v Running

# 2. Check resource utilization
echo "2. Checking resource utilization..."
kubectl top nodes
kubectl top pods -n hybrid-ai --sort-by=memory

# 3. Check database health
echo "3. Checking database health..."
kubectl exec -it postgres-primary-0 -n hybrid-ai -- \
  psql -U admin -d hybrid_ai -c "SELECT pg_database_size('hybrid_ai');"

# 4. Check model performance
echo "4. Checking model performance..."
curl -s http://monitoring.hybrid-ai.com/api/v1/query?query=avg\(model_accuracy_score\)

# 5. Check error rates
echo "5. Checking error rates..."
curl -s http://monitoring.hybrid-ai.com/api/v1/query?query=rate\(http_requests_total\{status\~\"5..\"\}\[5m\]\)

# 6. Check backup status
echo "6. Checking backup status..."
kubectl get cronjobs -n hybrid-ai

echo "=== Daily maintenance complete ==="
```

#### 2. Weekly Maintenance Tasks
```bash
#!/bin/bash
# weekly-maintenance.sh

echo "=== Weekly Maintenance Tasks ==="

# 1. Database maintenance
echo "1. Running database maintenance..."
kubectl exec -it postgres-primary-0 -n hybrid-ai -- \
  psql -U admin -d hybrid_ai -c "VACUUM ANALYZE;"

# 2. Log rotation and cleanup
echo "2. Cleaning up old logs..."
kubectl exec -it $(kubectl get pods -l app=ai-orchestrator -o jsonpath='{.items[0].metadata.name}') -n hybrid-ai -- \
  find /var/log -name "*.log" -mtime +7 -delete

# 3. Model performance review
echo "3. Reviewing model performance..."
python3 scripts/weekly-model-review.py

# 4. Security scan
echo "4. Running security scan..."
kubectl exec -it security-scanner-pod -n hybrid-ai -- \
  /usr/local/bin/security-scan.sh

# 5. Capacity planning review
echo "5. Reviewing capacity metrics..."
python3 scripts/capacity-planning-review.py

echo "=== Weekly maintenance complete ==="
```

#### 3. Monthly Maintenance Tasks
```bash
#!/bin/bash
# monthly-maintenance.sh

echo "=== Monthly Maintenance Tasks ==="

# 1. Full system backup
echo "1. Creating full system backup..."
kubectl create job --from=cronjob/full-backup full-backup-$(date +%Y%m%d) -n hybrid-ai

# 2. Security updates
echo "2. Applying security updates..."
kubectl set image deployment/ai-orchestrator \
  ai-orchestrator=hybrid-ai/orchestrator:latest-security

# 3. Performance optimization
echo "3. Running performance optimization..."
python3 scripts/performance-optimization.py

# 4. Compliance audit
echo "4. Running compliance audit..."
python3 scripts/compliance-audit.py

# 5. Disaster recovery test
echo "5. Testing disaster recovery procedures..."
python3 scripts/dr-test.py

echo "=== Monthly maintenance complete ==="
```

### Backup and Recovery Procedures

#### 1. Database Backup
```bash
# Create database backup
kubectl exec -it postgres-primary-0 -n hybrid-ai -- \
  pg_dump -U admin -d hybrid_ai -f /backups/hybrid_ai_$(date +%Y%m%d_%H%M%S).sql

# Verify backup integrity
kubectl exec -it postgres-primary-0 -n hybrid-ai -- \
  pg_restore --list /backups/hybrid_ai_$(date +%Y%m%d_%H%M%S).sql

# Upload to cloud storage
kubectl exec -it postgres-primary-0 -n hybrid-ai -- \
  aws s3 cp /backups/hybrid_ai_$(date +%Y%m%d_%H%M%S).sql \
  s3://hybrid-ai-backups/database/
```

#### 2. Model Artifact Backup
```bash
# Backup model artifacts
kubectl exec -it model-registry-pod -n hybrid-ai -- \
  tar -czf /backups/models_$(date +%Y%m%d_%H%M%S).tar.gz /models/

# Upload to cloud storage
kubectl exec -it model-registry-pod -n hybrid-ai -- \
  aws s3 cp /backups/models_$(date +%Y%m%d_%H%M%S).tar.gz \
  s3://hybrid-ai-backups/models/
```

#### 3. Configuration Backup
```bash
# Backup Kubernetes configurations
kubectl get all -n hybrid-ai -o yaml > k8s-backup-$(date +%Y%m%d).yaml

# Backup ConfigMaps and Secrets
kubectl get configmaps,secrets -n hybrid-ai -o yaml > config-backup-$(date +%Y%m%d).yaml

# Upload to version control
git add k8s-backup-$(date +%Y%m%d).yaml config-backup-$(date +%Y%m%d).yaml
git commit -m "Automated backup $(date +%Y%m%d)"
git push origin main
```

### Recovery Procedures

#### 1. Database Recovery
```bash
# Stop application services
kubectl scale deployment ai-orchestrator --replicas=0 -n hybrid-ai

# Restore database from backup
kubectl exec -it postgres-primary-0 -n hybrid-ai -- \
  psql -U admin -d hybrid_ai -f /backups/hybrid_ai_20240115_120000.sql

# Verify data integrity
kubectl exec -it postgres-primary-0 -n hybrid-ai -- \
  psql -U admin -d hybrid_ai -c "SELECT count(*) FROM analysis_results;"

# Restart application services
kubectl scale deployment ai-orchestrator --replicas=3 -n hybrid-ai
```

#### 2. Model Recovery
```bash
# Download model artifacts from backup
kubectl exec -it model-registry-pod -n hybrid-ai -- \
  aws s3 cp s3://hybrid-ai-backups/models/models_20240115_120000.tar.gz /tmp/

# Extract and restore models
kubectl exec -it model-registry-pod -n hybrid-ai -- \
  tar -xzf /tmp/models_20240115_120000.tar.gz -C /

# Restart model services
kubectl rollout restart deployment/earnings-analyzer -n hybrid-ai
kubectl rollout restart deployment/sentiment-analyzer -n hybrid-ai
```

## Module 5: Security Administration

### Access Control Management

#### 1. RBAC Configuration
```yaml
# rbac/admin-role.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: hybrid-ai-admin
rules:
- apiGroups: [""]
  resources: ["*"]
  verbs: ["*"]
- apiGroups: ["apps"]
  resources: ["*"]
  verbs: ["*"]
- apiGroups: ["extensions"]
  resources: ["*"]
  verbs: ["*"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: hybrid-ai-admin-binding
subjects:
- kind: User
  name: admin@hybrid-ai.com
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole
  name: hybrid-ai-admin
  apiGroup: rbac.authorization.k8s.io
```

#### 2. Service Account Management
```bash
# Create service account for monitoring
kubectl create serviceaccount monitoring-sa -n hybrid-ai

# Create role for monitoring
kubectl create role monitoring-role \
  --verb=get,list,watch \
  --resource=pods,services,endpoints \
  -n hybrid-ai

# Bind role to service account
kubectl create rolebinding monitoring-binding \
  --role=monitoring-role \
  --serviceaccount=hybrid-ai:monitoring-sa \
  -n hybrid-ai
```

### Security Monitoring

#### 1. Security Event Monitoring
```bash
# Monitor failed authentication attempts
kubectl logs -f deployment/api-gateway -n hybrid-ai | grep "authentication failed"

# Monitor unauthorized access attempts
kubectl logs -f deployment/api-gateway -n hybrid-ai | grep "unauthorized"

# Check for suspicious activity
kubectl exec -it prometheus-0 -n monitoring -- \
  promtool query instant 'rate(http_requests_total{status="401"}[5m])'
```

#### 2. Vulnerability Scanning
```bash
# Scan container images for vulnerabilities
kubectl exec -it security-scanner-pod -n hybrid-ai -- \
  trivy image hybrid-ai/orchestrator:latest

# Scan Kubernetes configurations
kubectl exec -it security-scanner-pod -n hybrid-ai -- \
  kube-bench run --targets master,node
```

### Certificate Management

#### 1. TLS Certificate Renewal
```bash
# Check certificate expiration
kubectl get certificates -n hybrid-ai

# Renew certificates
kubectl delete certificate api-tls -n hybrid-ai
kubectl apply -f k8s/certificates/api-tls.yaml

# Verify certificate renewal
kubectl describe certificate api-tls -n hybrid-ai
```

#### 2. Internal CA Management
```bash
# Generate new CA certificate
openssl genrsa -out ca-key.pem 4096
openssl req -new -x509 -days 365 -key ca-key.pem -out ca-cert.pem

# Update CA certificate in cluster
kubectl create secret tls ca-secret \
  --cert=ca-cert.pem \
  --key=ca-key.pem \
  -n hybrid-ai
```

## Module 6: Performance Optimization

### Resource Optimization

#### 1. CPU and Memory Optimization
```bash
# Analyze resource usage patterns
kubectl top pods -n hybrid-ai --sort-by=cpu
kubectl top pods -n hybrid-ai --sort-by=memory

# Optimize resource requests and limits
kubectl patch deployment ai-orchestrator -n hybrid-ai -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "ai-orchestrator",
            "resources": {
              "requests": {
                "cpu": "500m",
                "memory": "1Gi"
              },
              "limits": {
                "cpu": "2000m",
                "memory": "4Gi"
              }
            }
          }
        ]
      }
    }
  }
}'
```

#### 2. Auto-scaling Configuration
```yaml
# hpa/ai-orchestrator-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-orchestrator-hpa
  namespace: hybrid-ai
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-orchestrator
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Database Optimization

#### 1. Query Performance Tuning
```sql
-- Identify slow queries
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;

-- Add missing indexes
CREATE INDEX CONCURRENTLY idx_analysis_results_timestamp
ON analysis_results(created_at);

CREATE INDEX CONCURRENTLY idx_model_outputs_confidence
ON model_outputs(confidence_score);

-- Update table statistics
ANALYZE analysis_results;
ANALYZE model_outputs;
```

#### 2. Connection Pool Optimization
```yaml
# config/pgbouncer.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: pgbouncer-config
  namespace: hybrid-ai
data:
  pgbouncer.ini: |
    [databases]
    hybrid_ai = host=postgres-primary port=5432 dbname=hybrid_ai
    
    [pgbouncer]
    pool_mode = transaction
    max_client_conn = 1000
    default_pool_size = 50
    min_pool_size = 10
    reserve_pool_size = 5
    server_lifetime = 3600
    server_idle_timeout = 600
```

### Caching Optimization

#### 1. Redis Cache Optimization
```bash
# Monitor cache hit rate
kubectl exec -it redis-0 -n hybrid-ai -- \
  redis-cli info stats | grep keyspace_hits

# Optimize cache configuration
kubectl exec -it redis-0 -n hybrid-ai -- \
  redis-cli config set maxmemory-policy allkeys-lru

# Set optimal memory limit
kubectl exec -it redis-0 -n hybrid-ai -- \
  redis-cli config set maxmemory 2gb
```

#### 2. Application-Level Caching
```python
# Implement intelligent caching strategy
class CacheManager:
    def __init__(self):
        self.redis_client = redis.Redis(host='redis-service')
        self.cache_ttl = {
            'model_predictions': 3600,  # 1 hour
            'market_data': 300,         # 5 minutes
            'user_profiles': 1800       # 30 minutes
        }
    
    async def get_or_set(self, key: str, fetch_func, cache_type: str):
        # Try to get from cache first
        cached_value = await self.redis_client.get(key)
        if cached_value:
            return json.loads(cached_value)
        
        # Fetch and cache if not found
        value = await fetch_func()
        ttl = self.cache_ttl.get(cache_type, 3600)
        await self.redis_client.setex(key, ttl, json.dumps(value))
        return value
```

## Module 7: Troubleshooting and Problem Resolution

### Common Issues and Solutions

#### 1. High Response Time Issues
**Symptoms**: API responses taking >1000ms
**Diagnostic Steps**:
```bash
# Check system load
kubectl top nodes
kubectl top pods -n hybrid-ai

# Identify bottlenecks
kubectl exec -it prometheus-0 -n monitoring -- \
  promtool query instant 'histogram_quantile(0.95, http_request_duration_seconds_bucket)'

# Check database performance
kubectl exec -it postgres-primary-0 -n hybrid-ai -- \
  psql -U admin -d hybrid_ai -c "SELECT * FROM pg_stat_activity WHERE state = 'active';"
```

**Resolution Steps**:
```bash
# Scale up services
kubectl scale deployment ai-orchestrator --replicas=10 -n hybrid-ai

# Optimize database queries
kubectl exec -it postgres-primary-0 -n hybrid-ai -- \
  psql -U admin -d hybrid_ai -c "REINDEX DATABASE hybrid_ai;"

# Clear cache if corrupted
kubectl exec -it redis-0 -n hybrid-ai -- redis-cli flushdb
```

#### 2. Model Inference Failures
**Symptoms**: Models returning error responses
**Diagnostic Steps**:
```bash
# Check model pod status
kubectl get pods -l component=model -n hybrid-ai

# Check GPU utilization
kubectl exec -it earnings-analyzer-pod -n hybrid-ai -- nvidia-smi

# Test model endpoints
curl -X POST http://earnings-analyzer-service:8080/health
```

**Resolution Steps**:
```bash
# Restart failed models
kubectl delete pod -l app=earnings-analyzer -n hybrid-ai

# Scale model replicas
kubectl scale deployment earnings-analyzer --replicas=3 -n hybrid-ai

# Update model resources
kubectl patch deployment earnings-analyzer -n hybrid-ai -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "earnings-analyzer",
            "resources": {
              "limits": {
                "memory": "8Gi",
                "nvidia.com/gpu": 1
              }
            }
          }
        ]
      }
    }
  }
}'
```

### Emergency Procedures

#### 1. System Outage Response
```bash
#!/bin/bash
# emergency-response.sh

echo "=== EMERGENCY RESPONSE ACTIVATED ==="

# 1. Assess situation
kubectl get nodes
kubectl get pods --all-namespaces | grep -v Running

# 2. Enable maintenance mode
kubectl patch configmap system-config -n hybrid-ai --patch '
{
  "data": {
    "maintenance_mode": "true",
    "maintenance_message": "System maintenance in progress"
  }
}'

# 3. Check external dependencies
curl -I https://api.openai.com/v1/models
curl -I https://api.anthropic.com/v1/messages

# 4. Restore from backup if needed
if [ "$1" == "restore" ]; then
  echo "Restoring from backup..."
  kubectl exec -it postgres-primary-0 -n hybrid-ai -- \
    pg_restore -U admin -d hybrid_ai /backups/latest_backup.sql
fi

# 5. Gradual service restoration
kubectl rollout restart deployment/database -n hybrid-ai
sleep 60
kubectl rollout restart deployment/redis -n hybrid-ai
sleep 30
kubectl rollout restart deployment/ai-orchestrator -n hybrid-ai
sleep 30
kubectl rollout restart deployment/api-gateway -n hybrid-ai

echo "=== EMERGENCY RESPONSE COMPLETE ==="
```

## Module 8: Advanced Administration

### Custom Resource Definitions (CRDs)

#### 1. Model CRD Definition
```yaml
# crds/model-crd.yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: aimodels.hybrid-ai.com
spec:
  group: hybrid-ai.com
  versions:
  - name: v1
    served: true
    storage: true
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            properties:
              modelType:
                type: string
                enum: ["foundation", "specialized"]
              version:
                type: string
              resources:
                type: object
                properties:
                  cpu:
                    type: string
                  memory:
                    type: string
                  gpu:
                    type: integer
  scope: Namespaced
  names:
    plural: aimodels
    singular: aimodel
    kind: AIModel
```

#### 2. Model Operator Implementation
```python
# operators/model_operator.py
import kopf
import kubernetes

@kopf.on.create('hybrid-ai.com', 'v1', 'aimodels')
async def create_model(spec, name, namespace, **kwargs):
    """Handle AIModel creation"""
    
    # Create deployment for the model
    deployment = create_model_deployment(spec, name, namespace)
    
    # Create service for the model
    service = create_model_service(name, namespace)
    
    # Register model in registry
    await register_model_in_registry(name, spec)
    
    return {'message': f'Model {name} created successfully'}

@kopf.on.update('hybrid-ai.com', 'v1', 'aimodels')
async def update_model(spec, name, namespace, **kwargs):
    """Handle AIModel updates"""
    
    # Update deployment
    await update_model_deployment(spec, name, namespace)
    
    # Update model registry
    await update_model_in_registry(name, spec)
    
    return {'message': f'Model {name} updated successfully'}
```

### GitOps Integration

#### 1. ArgoCD Configuration
```yaml
# gitops/argocd-app.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: hybrid-ai
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/company/hybrid-ai-config
    targetRevision: HEAD
    path: k8s
  destination:
    server: https://kubernetes.default.svc
    namespace: hybrid-ai
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
    - CreateNamespace=true
```

#### 2. Flux Configuration
```yaml
# gitops/flux-kustomization.yaml
apiVersion: kustomize.toolkit.fluxcd.io/v1beta2
kind: Kustomization
metadata:
  name: hybrid-ai
  namespace: flux-system
spec:
  interval: 10m
  sourceRef:
    kind: GitRepository
    name: hybrid-ai-config
  path: "./k8s"
  prune: true
  validation: client
  healthChecks:
    - apiVersion: apps/v1
      kind: Deployment
      name: ai-orchestrator
      namespace: hybrid-ai
```

## Certification and Assessment

### Practical Exercises

#### Exercise 1: System Deployment
Deploy a complete Hybrid AI Architecture system from scratch:
1. Set up Kubernetes cluster
2. Deploy all core services
3. Configure monitoring and alerting
4. Verify system functionality

#### Exercise 2: Performance Optimization
Optimize a poorly performing system:
1. Identify performance bottlenecks
2. Implement optimization strategies
3. Measure improvement results
4. Document optimization procedures

#### Exercise 3: Disaster Recovery
Execute a complete disaster recovery scenario:
1. Simulate system failure
2. Execute recovery procedures
3. Verify data integrity
4. Document lessons learned

### Assessment Criteria

#### Technical Competency (70%)
- **System Architecture Understanding**: 20%
- **Deployment and Configuration**: 20%
- **Monitoring and Troubleshooting**: 15%
- **Performance Optimization**: 15%

#### Operational Excellence (30%)
- **Documentation Quality**: 10%
- **Problem-Solving Approach**: 10%
- **Communication Skills**: 10%

### Certification Levels

#### Level 1: Associate Administrator
- Basic system operation and monitoring
- Routine maintenance procedures
- Basic troubleshooting skills

#### Level 2: Professional Administrator
- Advanced configuration and optimization
- Complex troubleshooting and problem resolution
- Performance tuning and capacity planning

#### Level 3: Expert Administrator
- System architecture design and implementation
- Advanced automation and tooling
- Mentoring and training other administrators

This comprehensive training guide ensures system administrators have the knowledge and skills needed to effectively manage the Hybrid AI Architecture platform in production environments.
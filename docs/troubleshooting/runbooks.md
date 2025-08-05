# Hybrid AI Architecture - Troubleshooting Runbooks

## Overview

This document provides comprehensive troubleshooting procedures for the Hybrid AI Architecture system. Each runbook includes symptoms, root cause analysis, step-by-step resolution procedures, and prevention strategies.

## System Health Monitoring

### Health Check Endpoints
```bash
# Overall system health
curl -H "Authorization: Bearer $API_KEY" \
  https://api.hybrid-ai.platform.com/v1/system/health

# Component-specific health
curl -H "Authorization: Bearer $API_KEY" \
  https://api.hybrid-ai.platform.com/v1/system/health/components
```

### Key Health Indicators
- **Response Time**: <500ms for 95% of requests
- **Error Rate**: <1% sustained error rate
- **Model Availability**: >99.5% uptime for critical models
- **Resource Utilization**: <90% CPU/Memory usage
- **Queue Depth**: <100 pending requests

## Runbook 1: High Response Time

### Symptoms
- API response times >1000ms
- User complaints about slow performance
- Dashboard showing elevated P95/P99 latencies

### Diagnostic Steps

#### 1. Check System Metrics
```bash
# Check current response times
kubectl exec -it monitoring-pod -- \
  curl "http://prometheus:9090/api/v1/query?query=http_request_duration_seconds{quantile=\"0.95\"}"

# Check resource utilization
kubectl top nodes
kubectl top pods -n hybrid-ai
```

#### 2. Identify Bottlenecks
```bash
# Check database performance
kubectl exec -it postgres-pod -- \
  psql -U admin -d hybrid_ai -c "SELECT * FROM pg_stat_activity WHERE state = 'active';"

# Check Redis cache hit rate
kubectl exec -it redis-pod -- \
  redis-cli info stats | grep keyspace_hits
```

#### 3. Analyze Request Patterns
```bash
# Check request volume
kubectl logs -f api-gateway-pod | grep "request_count" | tail -100

# Identify slow endpoints
kubectl logs -f api-gateway-pod | grep "slow_request" | tail -50
```

### Resolution Steps

#### Immediate Actions (0-15 minutes)
1. **Scale API Pods**
   ```bash
   kubectl scale deployment api-gateway --replicas=10
   kubectl scale deployment ai-orchestrator --replicas=8
   ```

2. **Enable Request Throttling**
   ```bash
   kubectl patch configmap rate-limit-config --patch '{"data":{"rate_limit":"100"}}'
   kubectl rollout restart deployment api-gateway
   ```

3. **Clear Cache if Corrupted**
   ```bash
   kubectl exec -it redis-pod -- redis-cli flushdb
   ```

#### Short-term Actions (15-60 minutes)
1. **Optimize Database Queries**
   ```sql
   -- Identify slow queries
   SELECT query, mean_time, calls 
   FROM pg_stat_statements 
   ORDER BY mean_time DESC 
   LIMIT 10;
   
   -- Add missing indexes
   CREATE INDEX CONCURRENTLY idx_analysis_timestamp 
   ON analysis_results(timestamp);
   ```

2. **Scale Model Inference Pods**
   ```bash
   kubectl scale deployment earnings-analyzer --replicas=5
   kubectl scale deployment sentiment-analyzer --replicas=5
   kubectl scale deployment risk-predictor --replicas=5
   ```

3. **Enable Aggressive Caching**
   ```bash
   kubectl patch configmap cache-config --patch '{"data":{"ttl":"3600","max_size":"1GB"}}'
   ```

#### Long-term Actions (1-24 hours)
1. **Implement Connection Pooling**
   ```yaml
   # Update database configuration
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: postgres-config
   data:
     max_connections: "200"
     shared_buffers: "256MB"
     effective_cache_size: "1GB"
   ```

2. **Optimize Model Loading**
   ```python
   # Implement model warm-up
   async def warm_up_models():
       for model in specialized_models:
           await model.load_and_cache()
   ```

### Prevention Strategies
- **Auto-scaling**: Configure HPA based on response time metrics
- **Circuit Breakers**: Implement circuit breakers for external dependencies
- **Caching Strategy**: Implement multi-layer caching (Redis, CDN, application)
- **Database Optimization**: Regular query optimization and index maintenance

## Runbook 2: Model Inference Failures

### Symptoms
- Models returning error responses
- High error rates in model-specific endpoints
- "Model Unavailable" errors in logs

### Diagnostic Steps

#### 1. Check Model Health
```bash
# Check model pod status
kubectl get pods -l app=specialized-models

# Check model logs
kubectl logs -f earnings-analyzer-pod --tail=100
kubectl logs -f sentiment-analyzer-pod --tail=100
```

#### 2. Verify Model Resources
```bash
# Check GPU utilization
kubectl exec -it earnings-analyzer-pod -- nvidia-smi

# Check memory usage
kubectl top pods -l app=specialized-models
```

#### 3. Test Model Endpoints
```bash
# Test individual model health
curl -X POST http://earnings-analyzer-service:8080/health
curl -X POST http://sentiment-analyzer-service:8080/health
```

### Resolution Steps

#### Immediate Actions (0-5 minutes)
1. **Restart Failed Models**
   ```bash
   kubectl delete pod -l app=earnings-analyzer
   kubectl delete pod -l app=sentiment-analyzer
   ```

2. **Enable Fallback Models**
   ```bash
   kubectl patch configmap model-config --patch '{"data":{"fallback_enabled":"true"}}'
   ```

#### Short-term Actions (5-30 minutes)
1. **Scale Model Replicas**
   ```bash
   kubectl scale deployment earnings-analyzer --replicas=3
   kubectl scale deployment sentiment-analyzer --replicas=3
   ```

2. **Check Model Artifacts**
   ```bash
   # Verify model files
   kubectl exec -it earnings-analyzer-pod -- ls -la /models/
   
   # Re-download if corrupted
   kubectl exec -it earnings-analyzer-pod -- \
     aws s3 sync s3://model-artifacts/earnings-analyzer /models/
   ```

3. **Increase Resource Limits**
   ```yaml
   resources:
     limits:
       memory: "8Gi"
       nvidia.com/gpu: 1
     requests:
       memory: "4Gi"
       nvidia.com/gpu: 1
   ```

#### Long-term Actions (30+ minutes)
1. **Model Health Monitoring**
   ```python
   # Implement comprehensive health checks
   async def model_health_check():
       test_input = generate_test_input()
       result = await model.predict(test_input)
       assert result.confidence > 0.5
   ```

2. **Automated Model Recovery**
   ```bash
   # Create model recovery script
   #!/bin/bash
   if ! curl -f http://model-service:8080/health; then
       kubectl rollout restart deployment model-service
   fi
   ```

### Prevention Strategies
- **Health Checks**: Implement comprehensive model health monitoring
- **Resource Monitoring**: Monitor GPU memory and utilization
- **Model Versioning**: Maintain multiple model versions for rollback
- **Graceful Degradation**: Implement fallback to foundation models

## Runbook 3: Database Connection Issues

### Symptoms
- Database connection timeouts
- "Connection pool exhausted" errors
- Slow database queries

### Diagnostic Steps

#### 1. Check Database Status
```bash
# Check PostgreSQL pod status
kubectl get pods -l app=postgresql

# Check database connections
kubectl exec -it postgres-pod -- \
  psql -U admin -d hybrid_ai -c "SELECT count(*) FROM pg_stat_activity;"
```

#### 2. Analyze Connection Pool
```bash
# Check connection pool metrics
kubectl exec -it api-gateway-pod -- \
  curl http://localhost:8080/metrics | grep db_connections
```

#### 3. Review Database Performance
```sql
-- Check for long-running queries
SELECT pid, now() - pg_stat_activity.query_start AS duration, query 
FROM pg_stat_activity 
WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes';

-- Check for locks
SELECT * FROM pg_locks WHERE NOT granted;
```

### Resolution Steps

#### Immediate Actions (0-10 minutes)
1. **Kill Long-Running Queries**
   ```sql
   -- Terminate problematic queries
   SELECT pg_terminate_backend(pid) 
   FROM pg_stat_activity 
   WHERE (now() - pg_stat_activity.query_start) > interval '10 minutes';
   ```

2. **Increase Connection Limits**
   ```bash
   kubectl patch configmap postgres-config --patch \
     '{"data":{"max_connections":"300","shared_buffers":"512MB"}}'
   kubectl rollout restart statefulset postgresql
   ```

3. **Scale Database Replicas**
   ```bash
   kubectl scale statefulset postgresql-replica --replicas=2
   ```

#### Short-term Actions (10-60 minutes)
1. **Optimize Connection Pooling**
   ```yaml
   # Update connection pool configuration
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: pgbouncer-config
   data:
     pgbouncer.ini: |
       [databases]
       hybrid_ai = host=postgresql port=5432 dbname=hybrid_ai
       [pgbouncer]
       pool_mode = transaction
       max_client_conn = 1000
       default_pool_size = 50
   ```

2. **Add Database Monitoring**
   ```bash
   # Deploy PostgreSQL exporter
   kubectl apply -f monitoring/postgres-exporter.yaml
   ```

#### Long-term Actions (1+ hours)
1. **Database Performance Tuning**
   ```sql
   -- Optimize PostgreSQL configuration
   ALTER SYSTEM SET shared_buffers = '1GB';
   ALTER SYSTEM SET effective_cache_size = '3GB';
   ALTER SYSTEM SET maintenance_work_mem = '256MB';
   SELECT pg_reload_conf();
   ```

2. **Implement Read Replicas**
   ```yaml
   # Deploy read replica
   apiVersion: postgresql.cnpg.io/v1
   kind: Cluster
   metadata:
     name: postgres-replica
   spec:
     instances: 2
     primaryUpdateStrategy: unsupervised
   ```

### Prevention Strategies
- **Connection Monitoring**: Monitor connection pool utilization
- **Query Optimization**: Regular query performance analysis
- **Database Maintenance**: Automated vacuum and analyze operations
- **Capacity Planning**: Proactive scaling based on usage patterns

## Runbook 4: Guardrail System Failures

### Symptoms
- Guardrail validation errors
- High false positive rates
- Guardrail processing timeouts

### Diagnostic Steps

#### 1. Check Guardrail Service Health
```bash
# Check guardrail pod status
kubectl get pods -l app=guardrail-engine

# Check guardrail logs
kubectl logs -f guardrail-engine-pod --tail=100
```

#### 2. Analyze Validation Metrics
```bash
# Check validation success rates
kubectl exec -it monitoring-pod -- \
  curl "http://prometheus:9090/api/v1/query?query=guardrail_validation_success_rate"

# Check processing times
kubectl exec -it monitoring-pod -- \
  curl "http://prometheus:9090/api/v1/query?query=guardrail_processing_duration"
```

#### 3. Review Validation Rules
```bash
# Check current validation rules
kubectl get configmap guardrail-rules -o yaml
```

### Resolution Steps

#### Immediate Actions (0-5 minutes)
1. **Restart Guardrail Service**
   ```bash
   kubectl rollout restart deployment guardrail-engine
   ```

2. **Enable Bypass Mode**
   ```bash
   kubectl patch configmap guardrail-config --patch \
     '{"data":{"bypass_mode":"true","bypass_duration":"30m"}}'
   ```

#### Short-term Actions (5-30 minutes)
1. **Scale Guardrail Pods**
   ```bash
   kubectl scale deployment guardrail-engine --replicas=5
   ```

2. **Optimize Validation Rules**
   ```yaml
   # Update validation configuration
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: guardrail-rules
   data:
     hallucination_threshold: "0.1"
     fact_check_timeout: "5s"
     ethics_check_enabled: "true"
   ```

3. **Clear Validation Cache**
   ```bash
   kubectl exec -it redis-pod -- redis-cli del "guardrail:*"
   ```

#### Long-term Actions (30+ minutes)
1. **Implement Guardrail Monitoring**
   ```python
   # Add comprehensive guardrail metrics
   async def monitor_guardrail_performance():
       metrics = await guardrail_engine.get_metrics()
       if metrics.false_positive_rate > 0.1:
           await alert_manager.send_alert("High false positive rate")
   ```

2. **Optimize Validation Pipeline**
   ```python
   # Implement parallel validation
   async def validate_output_parallel(output):
       tasks = [
           validate_hallucination(output),
           validate_facts(output),
           validate_ethics(output)
       ]
       results = await asyncio.gather(*tasks)
       return combine_validation_results(results)
   ```

### Prevention Strategies
- **Validation Monitoring**: Monitor validation success rates and processing times
- **Rule Optimization**: Regular review and optimization of validation rules
- **Performance Testing**: Load testing of guardrail system
- **Fallback Mechanisms**: Implement graceful degradation for validation failures

## Runbook 5: Human-in-the-Loop System Issues

### Symptoms
- Expert review queue not updating
- Notification system failures
- Feedback not being processed

### Diagnostic Steps

#### 1. Check Review System Status
```bash
# Check review service pods
kubectl get pods -l app=human-review-system

# Check review queue status
kubectl exec -it redis-pod -- redis-cli llen "review_queue:high_priority"
```

#### 2. Verify Notification System
```bash
# Check notification service logs
kubectl logs -f notification-service-pod --tail=50

# Test notification endpoints
curl -X POST http://notification-service:8080/test-notification
```

#### 3. Analyze Feedback Processing
```bash
# Check feedback processing logs
kubectl logs -f feedback-processor-pod --tail=100

# Check feedback queue depth
kubectl exec -it redis-pod -- redis-cli llen "feedback_queue"
```

### Resolution Steps

#### Immediate Actions (0-10 minutes)
1. **Restart Review Services**
   ```bash
   kubectl rollout restart deployment human-review-system
   kubectl rollout restart deployment notification-service
   ```

2. **Clear Stuck Queues**
   ```bash
   kubectl exec -it redis-pod -- redis-cli del "review_queue:*"
   kubectl exec -it redis-pod -- redis-cli del "feedback_queue"
   ```

#### Short-term Actions (10-60 minutes)
1. **Scale Review System**
   ```bash
   kubectl scale deployment human-review-system --replicas=3
   kubectl scale deployment feedback-processor --replicas=2
   ```

2. **Reset Notification Channels**
   ```bash
   # Test email notifications
   kubectl exec -it notification-service-pod -- \
     python -c "from notifications import send_test_email; send_test_email()"
   
   # Test Slack notifications
   kubectl exec -it notification-service-pod -- \
     python -c "from notifications import send_test_slack; send_test_slack()"
   ```

#### Long-term Actions (1+ hours)
1. **Implement Review System Monitoring**
   ```python
   # Add comprehensive review system metrics
   async def monitor_review_system():
       queue_depth = await redis.llen("review_queue:high_priority")
       if queue_depth > 100:
           await alert_manager.send_alert("Review queue backlog")
   ```

2. **Optimize Feedback Processing**
   ```python
   # Implement batch feedback processing
   async def process_feedback_batch():
       feedback_batch = await get_feedback_batch(size=50)
       await model_updater.update_models(feedback_batch)
   ```

### Prevention Strategies
- **Queue Monitoring**: Monitor review queue depths and processing times
- **Notification Testing**: Regular testing of notification channels
- **Feedback Validation**: Validate feedback before processing
- **Expert Engagement**: Monitor expert participation and response times

## Emergency Procedures

### System-Wide Outage

#### Immediate Response (0-15 minutes)
1. **Activate Incident Response Team**
   ```bash
   # Send emergency alerts
   curl -X POST https://alerts.pagerduty.com/generic/2010-04-15/create_event.json \
     -d '{"service_key":"YOUR_SERVICE_KEY","event_type":"trigger","description":"System-wide outage"}'
   ```

2. **Enable Maintenance Mode**
   ```bash
   kubectl patch configmap system-config --patch \
     '{"data":{"maintenance_mode":"true","maintenance_message":"System maintenance in progress"}}'
   ```

3. **Check Infrastructure Status**
   ```bash
   # Check cluster health
   kubectl get nodes
   kubectl get pods --all-namespaces | grep -v Running
   
   # Check external dependencies
   curl -I https://api.openai.com/v1/models
   curl -I https://api.anthropic.com/v1/messages
   ```

#### Recovery Actions (15-60 minutes)
1. **Restore from Backup**
   ```bash
   # Restore database from latest backup
   kubectl exec -it postgres-pod -- \
     pg_restore -U admin -d hybrid_ai /backups/latest_backup.sql
   
   # Restore Redis data
   kubectl exec -it redis-pod -- \
     redis-cli --rdb /backups/redis_backup.rdb
   ```

2. **Gradual Service Restoration**
   ```bash
   # Restore services in order
   kubectl rollout restart deployment database
   kubectl rollout restart deployment redis
   kubectl rollout restart deployment ai-orchestrator
   kubectl rollout restart deployment api-gateway
   ```

### Data Corruption

#### Detection and Assessment
```bash
# Check data integrity
kubectl exec -it postgres-pod -- \
  psql -U admin -d hybrid_ai -c "SELECT pg_database_size('hybrid_ai');"

# Verify model artifacts
kubectl exec -it model-storage-pod -- \
  find /models -name "*.pkl" -exec file {} \;
```

#### Recovery Procedures
```bash
# Restore from point-in-time backup
kubectl exec -it postgres-pod -- \
  pg_restore -U admin -d hybrid_ai_recovery /backups/backup_2024_01_15.sql

# Verify data integrity after restore
kubectl exec -it postgres-pod -- \
  psql -U admin -d hybrid_ai_recovery -c "SELECT count(*) FROM analysis_results;"
```

## Monitoring and Alerting

### Critical Alerts
- **System Down**: Any core service unavailable >5 minutes
- **High Error Rate**: >5% error rate sustained >10 minutes
- **Performance Degradation**: >2x normal response time >15 minutes
- **Data Loss**: Any indication of data corruption or loss
- **Security Breach**: Unauthorized access attempts

### Alert Escalation
1. **Level 1**: On-call engineer (immediate)
2. **Level 2**: Engineering manager (15 minutes)
3. **Level 3**: VP Engineering (30 minutes)
4. **Level 4**: CTO (60 minutes)

### Post-Incident Procedures
1. **Incident Documentation**: Complete incident report within 24 hours
2. **Root Cause Analysis**: Detailed RCA within 48 hours
3. **Action Items**: Preventive measures identified and scheduled
4. **Process Improvement**: Update runbooks and procedures
5. **Team Review**: Post-mortem meeting with all stakeholders

This comprehensive runbook ensures rapid resolution of system issues and maintains high availability of the Hybrid AI Architecture platform.
# MLOps Production Deployment Checklist

## Pre-Deployment

### Infrastructure
- [ ] VPC and networking configured (public/private subnets)
- [ ] Security groups configured with least privilege access
- [ ] Load balancers configured and tested
- [ ] DNS records created and verified
- [ ] SSL/TLS certificates provisioned
- [ ] CDN configured for static assets (if applicable)

### Data Storage
- [ ] PostgreSQL RDS instance provisioned with Multi-AZ
- [ ] Database backups configured (automated daily)
- [ ] ElastiCache Redis cluster provisioned
- [ ] S3 buckets created for artifacts and DVC storage
- [ ] S3 versioning enabled
- [ ] S3 lifecycle policies configured
- [ ] Backup and disaster recovery tested

### Secrets Management
- [ ] All passwords generated (strong, unique)
- [ ] Secrets stored in AWS Secrets Manager or HashiCorp Vault
- [ ] `.env` file configured with production values
- [ ] Kubernetes secrets created and encrypted
- [ ] IAM roles and policies configured
- [ ] Service accounts created with minimal permissions

### Monitoring & Logging
- [ ] Prometheus configured and accessible
- [ ] Grafana dashboards imported
- [ ] Alert rules configured
- [ ] Alert channels configured (Slack, PagerDuty, email)
- [ ] Log aggregation configured (CloudWatch, ELK, etc.)
- [ ] Log retention policies set
- [ ] Application instrumentation verified

### Security
- [ ] Security audit completed
- [ ] Vulnerability scanning configured
- [ ] Network security policies applied
- [ ] Firewall rules configured
- [ ] DDoS protection enabled
- [ ] WAF rules configured
- [ ] Encryption at rest enabled
- [ ] Encryption in transit enforced
- [ ] Authentication and authorization configured
- [ ] Rate limiting configured

## Deployment

### Docker Compose Deployment
- [ ] `.env` file configured
- [ ] Docker Compose file reviewed
- [ ] Images pulled successfully
- [ ] Volumes configured correctly
- [ ] Networks configured
- [ ] Services started: `docker-compose up -d`
- [ ] Health checks passing
- [ ] Services accessible

### Kubernetes Deployment
- [ ] Namespace created
- [ ] ConfigMaps applied
- [ ] Secrets applied
- [ ] PersistentVolumeClaims provisioned
- [ ] StatefulSets deployed (PostgreSQL)
- [ ] Deployments applied (MLFlow, Airflow, Dashboard)
- [ ] Services exposed
- [ ] Ingress configured
- [ ] HPA configured
- [ ] Resource limits set
- [ ] All pods running
- [ ] Health checks passing

### Terraform Infrastructure
- [ ] Terraform state backend configured (S3 + DynamoDB)
- [ ] Variables file configured (`terraform.tfvars`)
- [ ] Terraform plan reviewed
- [ ] Terraform apply executed successfully
- [ ] Outputs verified
- [ ] Resources tagged correctly
- [ ] Cost estimation reviewed

## Post-Deployment

### Validation
- [ ] All services accessible via public URLs
- [ ] MLFlow tracking server functional
- [ ] Airflow web UI accessible
- [ ] Dashboard accessible and displaying data
- [ ] Grafana dashboards showing metrics
- [ ] Database connections working
- [ ] Redis cache operational
- [ ] Object storage accessible
- [ ] Model serving endpoint responsive

### Testing
- [ ] Smoke tests passed
- [ ] Integration tests passed
- [ ] Performance tests passed
- [ ] Load tests passed
- [ ] Failover tests passed
- [ ] Backup restoration tested
- [ ] Disaster recovery procedure validated

### Documentation
- [ ] Deployment guide updated
- [ ] Runbook created
- [ ] Architecture diagrams updated
- [ ] API documentation published
- [ ] User guides updated
- [ ] Team onboarding materials updated

### Operational
- [ ] On-call rotation configured
- [ ] Incident response plan documented
- [ ] Escalation procedures defined
- [ ] SLA/SLO targets defined
- [ ] Capacity planning completed
- [ ] Cost monitoring configured
- [ ] Scheduled maintenance windows defined

## Monitoring & Maintenance

### Daily
- [ ] Check service health dashboard
- [ ] Review error logs
- [ ] Verify backup completion
- [ ] Monitor resource usage
- [ ] Check alert status

### Weekly
- [ ] Review performance metrics
- [ ] Analyze cost trends
- [ ] Review security alerts
- [ ] Update documentation if needed
- [ ] Clean up unused resources

### Monthly
- [ ] Security patches applied
- [ ] Dependency updates reviewed
- [ ] Capacity planning review
- [ ] Disaster recovery drill
- [ ] Cost optimization review
- [ ] Performance optimization review

## Rollback Plan

### Preparation
- [ ] Previous version artifacts available
- [ ] Rollback procedure documented
- [ ] Database migration rollback scripts prepared
- [ ] Communication plan for rollback

### Execution
- [ ] Stop new deployments
- [ ] Switch traffic to previous version
- [ ] Verify services operational
- [ ] Notify stakeholders
- [ ] Investigate root cause

## Sign-Off

### Technical Sign-Off
- [ ] DevOps Engineer: _______________
- [ ] Security Engineer: _______________
- [ ] Platform Lead: _______________

### Business Sign-Off
- [ ] Product Owner: _______________
- [ ] Engineering Manager: _______________

### Date
- Deployment Date: _______________
- Sign-Off Date: _______________

## Notes

### Known Issues
- List any known issues or limitations

### Deferred Items
- List items deferred to future releases

### Contacts
- On-Call Engineer: _______________
- Platform Team Lead: _______________
- Security Contact: _______________
- AWS Support: _______________

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.mlops.id
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = aws_subnet.private[*].id
}

output "rds_endpoint" {
  description = "RDS PostgreSQL endpoint"
  value       = aws_db_instance.mlops.endpoint
}

output "rds_database_name" {
  description = "RDS database name"
  value       = aws_db_instance.mlops.db_name
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = aws_elasticache_cluster.mlops.cache_nodes[0].address
}

output "mlflow_artifacts_bucket" {
  description = "S3 bucket for MLFlow artifacts"
  value       = aws_s3_bucket.mlflow_artifacts.id
}

output "dvc_storage_bucket" {
  description = "S3 bucket for DVC storage"
  value       = aws_s3_bucket.dvc_storage.id
}

output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = var.enable_eks ? aws_eks_cluster.mlops[0].endpoint : null
}

output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = var.enable_eks ? aws_eks_cluster.mlops[0].name : null
}

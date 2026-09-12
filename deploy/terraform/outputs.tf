# ==============================================================================
# Output Variables
# ==============================================================================
output "ec2_public_ip" {
  value       = aws_eip.ec2_eip.public_ip
  description = "The public IP of the EC2 instance"
}

output "ec2_instance_id" {
  value       = aws_instance.web_server.id
  description = "The ID of the EC2 instance"
}

output "ecr_repository_url" {
  value       = aws_ecr_repository.app_repo.repository_url
  description = "The URL of the ECR repository"
}

output "s3_bucket_name" {
  value       = aws_s3_bucket.mlflow_bucket.id
  description = "The name of the S3 bucket for MLflow artifacts"
}

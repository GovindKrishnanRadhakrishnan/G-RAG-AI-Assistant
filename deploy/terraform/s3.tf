# ==============================================================================
# S3 Bucket Configuration (MLflow Artifacts - Free-Tier Eligible)
# ==============================================================================
resource "aws_s3_bucket" "mlflow_bucket" {
  bucket        = "rag-research-assistant-mlflow-storage-${var.environment}"
  force_destroy = true # Allows clean terraform destroy

  tags = {
    Name = "mlflow-artifact-bucket"
  }
}

# Block Public Access (MLOps best practice)
resource "aws_s3_bucket_public_access_block" "mlflow_bucket_public_block" {
  bucket = aws_s3_bucket.mlflow_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Enable Server-Side Encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "mlflow_bucket_encryption" {
  bucket = aws_s3_bucket.mlflow_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Enable Versioning
resource "aws_s3_bucket_versioning" "mlflow_bucket_versioning" {
  bucket = aws_s3_bucket.mlflow_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Allow EC2 Instance to read/write to the MLflow bucket
resource "aws_iam_policy" "ec2_s3_policy" {
  name        = "${var.environment}-ec2-s3-policy"
  description = "Allows EC2 instance to read/write to S3 MLflow bucket"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = [aws_s3_bucket.mlflow_bucket.arn]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = ["${aws_s3_bucket.mlflow_bucket.arn}/*"]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ec2_s3_attach" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = aws_iam_policy.ec2_s3_policy.arn
}

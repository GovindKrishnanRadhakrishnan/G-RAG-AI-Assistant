# ==============================================================================
# Elastic Container Registry (ECR) Configuration
# ==============================================================================
resource "aws_ecr_repository" "app_repo" {
  name                 = "rag-research-assistant"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Keep only the last 3 images to prevent storage build-up costs
resource "aws_ecr_lifecycle_policy" "repo_policy" {
  repository = aws_ecr_repository.app_repo.name

  policy = <<EOF
{
    "rules": [
        {
            "rulePriority": 1,
            "description": "Keep only last 3 images",
            "selection": {
                "tagStatus": "any",
                "countType": "imageCountMoreThan",
                "countNumber": 3
            },
            "action": {
                "type": "expire"
            }
        }
    ]
}
EOF
}

# ==============================================================================
# IAM Roles & Profiles Configuration
# ==============================================================================

# 1. EC2 Instance Role
resource "aws_iam_role" "ec2_role" {
  name = "${var.environment}-rag-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

# Attach AmazonEC2ContainerRegistryReadOnly to EC2 Instance
resource "aws_iam_role_policy_attachment" "ec2_ecr_policy" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

# IAM Instance Profile for EC2
resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${var.environment}-rag-ec2-profile"
  role = aws_iam_role.ec2_role.name
}

# 2. Jenkins CI/CD IAM Role (Used by Jenkins agent to push ECR images)
resource "aws_iam_role" "jenkins_role" {
  name = "${var.environment}-jenkins-ecr-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          # Restrict to your Jenkins agent role or AWS account
          AWS = "*"
        }
        Condition = {
          StringEquals = {
            "aws:PrincipalOrgID" = [] # Populate if using AWS Organizations
          }
        }
      }
    ]
  })
}

# Jenkins ECR Policy
resource "aws_iam_policy" "jenkins_ecr_policy" {
  name        = "${var.environment}-jenkins-ecr-policy"
  description = "Allows Jenkins to push images to ECR"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:GetRepositoryPolicy",
          "ecr:DescribeRepositories",
          "ecr:ListImages",
          "ecr:DescribeImages",
          "ecr:BatchGetImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:PutImage"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "jenkins_ecr_attach" {
  role       = aws_iam_role.jenkins_role.name
  policy_arn = aws_iam_policy.jenkins_ecr_policy.arn
}

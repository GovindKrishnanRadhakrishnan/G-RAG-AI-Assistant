# ==============================================================================
# Terraform Core Configuration & Providers
# ==============================================================================
terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  # Local state for zero-cost execution.
  # Can be migrated to S3 + DynamoDB for team environments.
  backend "local" {
    path = "terraform.tfstate"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Environment = var.environment
      Project     = "RAG-Research-Assistant"
      ManagedBy   = "Terraform"
    }
  }
}

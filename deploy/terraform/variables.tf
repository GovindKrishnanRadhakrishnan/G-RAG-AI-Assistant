# ==============================================================================
# Input Variables
# ==============================================================================
variable "aws_region" {
  type        = string
  description = "AWS region for provisioning resources"
  default     = "us-east-1"
}

variable "environment" {
  type        = string
  description = "Deployment environment"
  default     = "dev"
}

variable "instance_type" {
  type        = string
  description = "EC2 instance type (free-tier eligible)"
  default     = "t2.micro"
}

variable "my_ip" {
  type        = string
  description = "Your IP CIDR block for SSH access (default is wide open, restrict in production)"
  default     = "0.0.0.0/0"
}

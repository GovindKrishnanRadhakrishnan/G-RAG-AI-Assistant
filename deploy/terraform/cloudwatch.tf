# ==============================================================================
# CloudWatch Logging Configuration
# ==============================================================================

# CloudWatch Log Group for Application Logs
resource "aws_cloudwatch_log_group" "app_logs" {
  name              = "/ecs/rag-research-assistant-${var.environment}"
  retention_in_days = 7 # Cost saving: keep logs only for 7 days

  tags = {
    Name = "rag-app-log-group"
  }
}

# IAM Policy for EC2 to write to CloudWatch Logs
resource "aws_iam_policy" "ec2_cw_policy" {
  name        = "${var.environment}-ec2-cw-policy"
  description = "Allows EC2 instance to send logs to CloudWatch"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = ["${aws_cloudwatch_log_group.app_logs.arn}:*"]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ec2_cw_attach" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = aws_iam_policy.ec2_cw_policy.arn
}

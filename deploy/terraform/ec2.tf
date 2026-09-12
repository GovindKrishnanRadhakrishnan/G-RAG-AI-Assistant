# ==============================================================================
# EC2 Instance Configuration (t2.micro Free-Tier)
# ==============================================================================

# Security Group for RAG Stack
resource "aws_security_group" "ec2_sg" {
  name        = "${var.environment}-ec2-sg"
  description = "Security group for RAG application on EC2"
  vpc_id      = aws_vpc.main_vpc.id

  # SSH
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
    description = "SSH access"
  }

  # HTTP (FastAPI / Streamlit / Nginx Reverse Proxy)
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTP access"
  }

  # FastAPI Backend Port
  ingress {
    from_port   = 8001
    to_port     = 8001
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "FastAPI Backend"
  }

  # Streamlit Frontend Port
  ingress {
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Streamlit Frontend"
  }

  # Outbound internet access
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound"
  }

  tags = {
    Name = "${var.environment}-ec2-sg"
  }
}

# Fetch the latest Ubuntu 22.04 LTS AMI
data "aws_ami" "ubuntu" {
  most_recent = true
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
  owners = ["099720109477"] # Canonical
}

# EC2 Instance
resource "aws_instance" "web_server" {
  ami                  = data.aws_ami.ubuntu.id
  instance_type        = var.instance_type
  subnet_id            = aws_subnet.public_subnet.id
  vpc_security_group_ids = [aws_security_group.ec2_sg.id]
  iam_instance_profile = aws_iam_instance_profile.ec2_profile.name

  # Provision 30GB of gp3 storage (Free-tier max)
  root_block_device {
    volume_size           = 30
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }

  # Install Docker and Docker Compose on Startup
  user_data = <<-EOF
              #!/bin/bash
              apt-get update
              apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release awscli
              
              # Install Docker
              mkdir -p /etc/apt/keyrings
              curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
              echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
              apt-get update
              apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
              
              systemctl start docker
              systemctl enable docker
              usermod -aG docker ubuntu
              
              # Install Docker Compose Standalone
              curl -SL https://github.com/docker/compose/releases/download/v2.20.2/docker-compose-linux-x86_64 -o /usr/local/bin/docker-compose
              chmod +x /usr/local/bin/docker-compose
              EOF

  tags = {
    Name = "${var.environment}-rag-server"
  }
}

# Elastic IP for persistent public IP address
resource "aws_eip" "ec2_eip" {
  instance = aws_instance.web_server.id
  domain   = "vpc"

  tags = {
    Name = "${var.environment}-ec2-eip"
  }
}

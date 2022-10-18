/**
 * # 🤗 on Trainium (Infrastructure)
 *
 * ## Overview
 *
 * Terraform configuration for setting up an [🤗 on Trainium](https://julsimon.substack.com/p/video-accelerate-transformer-training-c49) demo.
 * 
 * ## Setup
 *
 * 0. Download and install [Terraform](https://www.terraform.io/)
 *
 * 1. Provision resources by running the following commands
 *
 *      > By default, this blueprint uses a [partial configuration](https://www.terraform.io/language/settings/backends/configuration#partial-configuration) to set up the [S3 backend](https://www.terraform.io/language/settings/backends/s3).
 *
 *        ```bash
 *        terraform init -upgrade -backend-config="config.s3.tfbackend"
 *        terraform plan
 *        terraform apply
 *        ```
 */

terraform {
  # Specifies which versions of Terraform can be used w/ this configuration
  required_version = ">= 1.2.1"

  required_providers {
    # Terraform provider for AWS
    # https://github.com/hashicorp/terraform-provider-aws
    aws = {
      source  = "hashicorp/aws"
      version = ">= 4.19.0"
    }
  }

  backend "s3" {
    # Uses a partial configuration
    # https://www.terraform.io/language/settings/backends/configuration#partial-configuration
    # to define an S3 backend with state locking and consistency checking via DynamoDB
    # https://www.terraform.io/language/settings/backends/s3
  }
}

provider "aws" {
  region = var.region

  # These tags will be inherited by all resources that support tags
  # https://www.hashicorp.com/blog/default-tags-in-the-terraform-aws-provider
  default_tags {
    tags = {
      Project     = var.project
      Environment = var.environment
      Owner       = var.owner
    }
  }
}

// Data

data "aws_ami" "amazon_linux_2" {
  most_recent = true


  filter {
    name   = "owner-alias"
    values = ["amazon"]
  }


  filter {
    name   = "name"
    values = ["amzn2-ami-hvm*"]
  }
}

resource "aws_security_group" "trainium" {
  name = "trainium_sg"

  ingress {
    description = "Allow SSH inbound traffic for EC2 Instance Connect"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [
      "0.0.0.0/0"
    ]
  }

  egress {
    from_port = 0
    to_port   = 0
    protocol  = "-1"
    cidr_blocks = [
      "0.0.0.0/0"
    ]
  }
}

resource "aws_instance" "trainium" {
  ami           = data.aws_ami.amazon_linux_2.id
  instance_type = var.instance_type
  user_data     = file("userdata.sh")

  # Force recreate on userdata change
  # https://github.com/hashicorp/terraform-provider-aws/issues/23315
  user_data_replace_on_change = true

  security_groups = [
    aws_security_group.trainium.name
  ]

  root_block_device {
    volume_size = 50
  }

  tags = {
    Name = "trainium-${var.owner}"
  }
}

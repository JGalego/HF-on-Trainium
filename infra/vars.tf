// vars.tf

variable "owner" {
  type        = string
  description = "The owner of the project"
}

variable "project" {
  type        = string
  description = "The name of the project"
  default     = "hf-trainium"

  validation {
    condition     = length(var.project) >= 3
    error_message = "Project name should be at least 3 characters long."
  }
}

variable "environment" {
  type        = string
  description = "The name of the environment"
  default     = "dev"
  validation {
    condition = contains([
      "dev",
      "test",
      "stage",
      "prod"
    ], var.environment)
    error_message = "Invalid environment name - Expected: [dev, test, stage, prod]."
  }
}

# For more information, see
# https://aws.amazon.com/about-aws/whats-new/2022/10/ec2-trn1-instances-high-performance-cost-effective-deep-learning-training/
variable "region" {
  type        = string
  description = "The default AWS region"
  default     = "us-east-1"

  validation {
    condition     = can(regex("[a-z][a-z]-[a-z]+-[1-9]", var.region))
    error_message = "Must be a valid AWS Region name."
  }

  validation {
    condition = contains([
      "us-east-1",
      "us-west-2"
    ], var.region)
    error_message = "As of October 2022, only us-east-1 (N. Virginia) and us-west-2 (Oregon) support AWS Trainium instances."
  }
}

# For more information, see
# https://aws.amazon.com/ec2/instance-types/trn1/
variable "instance_type" {
  type        = string
  description = "The instance size"
  default     = "trn1.2xlarge"

  validation {
    condition = contains([
      "trn1.2xlarge",
      "trn1.32xlarge"
    ], var.instance_type)
    error_message = "As of October 2022, only trn1.2xlarge and trn1.32xlarge instance sizes are supported."
  }
}
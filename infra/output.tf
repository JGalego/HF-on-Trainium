// output.tf

output "trainium_instance" {
  description = "AWS Trainium instance ID"
  value       = aws_instance.trainium.id
}

output "region" {
  description = "The AWS Region used"
  value = var.region
}
# 🤗 on Trainium (Infrastructure)

## Overview

Terraform configuration for setting up an [🤗 on Trainium](https://julsimon.substack.com/p/video-accelerate-transformer-training-c49) demo.

## Setup

0. Download and install [Terraform](https://www.terraform.io/)

1. Provision resources by running the following commands

     > By default, this blueprint uses a [partial configuration](https://www.terraform.io/language/settings/backends/configuration#partial-configuration) to set up the [S3 backend](https://www.terraform.io/language/settings/backends/s3).

      ```bash
      terraform init -upgrade -backend-config="config.s3.tfbackend"
      terraform plan
      terraform apply
      ```

## Providers

| Name | Version |
|------|---------|
| <a name="provider_aws"></a> [aws](#provider_aws) | 5.7.0 |

## Inputs

| Name | Description | Type | Default |
|------|-------------|------|---------|
| <a name="input_owner"></a> [owner](#input_owner) | The owner of the project | `string` | n/a |
| <a name="input_environment"></a> [environment](#input_environment) | The name of the environment | `string` | `"dev"` |
| <a name="input_instance_type"></a> [instance_type](#input_instance_type) | The instance size | `string` | `"trn1.2xlarge"` |
| <a name="input_project"></a> [project](#input_project) | The name of the project | `string` | `"hf-trainium"` |
| <a name="input_region"></a> [region](#input_region) | The default AWS region | `string` | `"us-east-1"` |

## Outputs

| Name | Description |
|------|-------------|
| <a name="output_region"></a> [region](#output_region) | The AWS Region used |
| <a name="output_trainium_instance"></a> [trainium_instance](#output_trainium_instance) | AWS Trainium instance ID |
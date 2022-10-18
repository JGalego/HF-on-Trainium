# 🤗 on Trainium

## Overview

A (not so deep) exploration of [🤗 Transformers](https://huggingface.co/docs/transformers/index) training on [AWS Trainium](https://aws.amazon.com/machine-learning/trainium/).

> Inspired by Julien Simon's post on how to [Accelerate Transformer training with AWS Trainium](https://julsimon.medium.com/accelerate-transformer-training-with-aws-trainium-d20cd3f9dc08) 🙌

<img src="hf_on_trainium.png">

## What is AWS Trainium?

> **Update (October 2022)** 📢 [Amazon EC2 `Trn1` instances powered by AWS-designed Trainium chips are now generally available](https://press.aboutamazon.com/news-releases/news-release-details/aws-announces-general-availability-amazon-ec2-trn1-instances)

AWS Trainium is a 2nd generation ML chip optimized for training state-of-the-art models.

Trainium-powered [Amazon EC2 `Trn1` instances](https://aws.amazon.com/ec2/instance-types/trn1/) achieve the highest performance on deep learning training, while providing ***up to 50% lower*** cost-to-train savings over comparable GPU-based `P4d` instances.

Using the [AWS Neuron SDK](https://aws.amazon.com/machine-learning/neuron/), which integrates with popular frameworks like TensorFlow, PyTorch and Apache MXNet, anyone can start using AWS Trainium by changing just a few lines of code.

![AWS Trainium](https://d2908q01vomqb2.cloudfront.net/da4b9237bacccdf19c0760cab7aec4a8359010b0/2022/08/19/Site-Merch_EC2-Trainium_Blog.png)

## Setup

1. Provision resources using [Terraform](https://www.terraform.io/).

	```bash
	# It will take a few minutes for the Trn1 instance to be fully configured ⌛
	cd infra
	terraform init -upgrade -backend-config="config.s3.tfbackend"
	terraform plan
	terraform apply

	# ⚠️ Clean up after yourself - don't forget to destroy all resources when you're done!
	terraform destroy
	```

2. Connect to the trainium instance using [EC2 Instance Connect](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Connect-using-EC2-Instance-Connect.html).

	```bash
	# For information on how to set up EC2 instance connect, see
	# https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-connect-set-up.html
	mssh -r $(terraform output -raw region) ec2-user@$(terraform output -raw trainium_instance)
	```

3. Run training job.

	```bash
	# Clone this repository to access the training scripts
	git clone https://github.com/JGalego/hf-on-trainium
	cd hf-on-trainium/examples

	# Install dependencies
	python3 -m pip install -r requirements.txt

	# CPU/GPU 🐌
	python3 original.py

	# Trainium (Single Core) ⚡
	python3 trainium_single.py

	# Trainium (Distributed) ⚡⚡⚡
	export TOKENIZERS_PARALLELISM=false  # disabling parallelism to avoid hidden deadlocks
	export N_PROCS_PER_NODE=2  			 # either 1, 2, 8 or a multiple of 32
	torchrun --nproc_per_node=$N_PROCS_PER_NODE trainium_distributed.py

	# Track Neuron environment activity
	neuron-top
	```

## References

* [AWS Trainium](https://aws.amazon.com/machine-learning/trainium/)
* [Amazon EC2 `Trn1` instances](https://aws.amazon.com/ec2/instance-types/trn1/)
* [AWS Neuron SDK documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/index.html)
* [AWS Neuron SDK samples](https://github.com/aws-neuron/aws-neuron-samples)
* [PyTorch on XLA devices](https://pytorch.org/xla/release/1.11/index.html)
* [AWS On Air feat. Silicon Innovation: Trainium and Inferentia](https://www.youtube.com/watch?v=vVanYs0h1bw)
* [Amazon EC2 Trn1 Instances for High-Performance Model Training are Now Available](https://aws.amazon.com/blogs/aws/amazon-ec2-trn1-instances-for-high-performance-model-training-are-now-available/)
* (HuggingFace) [Tutorial: Fine-tune a pretrained model](https://huggingface.co/docs/transformers/training)
* (Julien Simon) [Video: Accelerate Transformer training with AWS Trainium](https://julsimon.medium.com/accelerate-transformer-training-with-aws-trainium-d20cd3f9dc08)
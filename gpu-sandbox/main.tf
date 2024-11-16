terraform {
  backend "s3" {
    bucket = "seafoodfry-tf-backend"
    key    = "gpu-sandbox"
    region = "us-east-2"
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.76"
    }
  }

  required_version = "~> 1.9.8"
}

provider "aws" {
  region = "us-east-2"
}

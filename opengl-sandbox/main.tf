terraform {
  backend "s3" {
    bucket = "seafoodfry-tf-backend"
    key    = "opengl-sandbox"
    region = "us-east-2"
    }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.5"
    }
  }

  required_version = "~> 1.8.4"
}

provider "aws" {
  region = "us-east-2"
}

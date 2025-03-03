terraform {
  backend "s3" {
    bucket = "seafoodfry-tf-backend"
    key    = "gpu-sandbox"
    region = "us-east-2"
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.89"
    }
  }

  required_version = ">= 1.11"
}

provider "aws" {
  alias  = "us_east_2"
  region = "us-east-2"
}

provider "aws" {
  #alias = "us_east_1"
  region = "us-east-1"
}
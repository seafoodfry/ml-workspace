locals {
  region_azs = {
    "us-east-2" = ["us-east-2a", "us-east-2b", "us-east-2c"]
    "us-east-1" = ["us-east-1a", "us-east-1b", "us-east-1c"]
  }

  selected_azs = lookup(local.region_azs, data.aws_region.current.name, [])
}

module "vpc" {
  count   = 1
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.19"

  name = "sandbox"
  cidr = "10.0.0.0/16"

  azs             = local.selected_azs
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]


  enable_nat_gateway     = true
  single_nat_gateway     = true
  one_nat_gateway_per_az = false

}

resource "aws_security_group" "ssh" {
  count = length(module.vpc) > 0 ? 1 : 0

  name        = "ssh"
  description = "Allow SSH from a specific IP"
  vpc_id      = module.vpc[0].vpc_id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["${var.my_ip}/32"]
  }

  egress {
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }
}

resource "aws_security_group" "ssh_and_mpi" {
  count = length(module.vpc) > 0 ? 1 : 0

  name        = "ssh-mpi"
  description = "Allow SSH from a specific IP and MPI within nodes"
  vpc_id      = module.vpc[0].vpc_id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["${var.my_ip}/32"]
  }

  # Allow all internal communication for MPI.
  # NOTE: the self argument is marked true!!!
  ingress {
    from_port = 0
    to_port   = 0
    protocol  = "-1"
    self      = true # NOTE: should always be true!!!
  }

  egress {
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }
}

resource "aws_security_group" "ssh_and_dcv" {
  count = length(module.vpc) > 0 ? 1 : 0

  name        = "ssh_and_dcv"
  description = "Allow SSH and NICE DCV from a specific IP"
  vpc_id      = module.vpc[0].vpc_id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["${var.my_ip}/32"]
  }
  ingress {
    description = "DCV"
    from_port   = 8443
    to_port     = 8443
    protocol    = "tcp"
    cidr_blocks = ["${var.my_ip}/32"]
  }

  egress {
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }
}


resource "aws_security_group" "rdp" {
  count = length(module.vpc) > 0 ? 1 : 0

  name        = "rdp"
  description = "Allow RDP from a specific IP"
  vpc_id      = module.vpc[0].vpc_id

  ingress {
    description = "RDP"
    from_port   = 3389
    to_port     = 3389
    protocol    = "tcp"
    cidr_blocks = ["${var.my_ip}/32"]
  }
  ingress {
    description = "DCV"
    from_port   = 8443
    to_port     = 8443
    protocol    = "tcp"
    cidr_blocks = ["${var.my_ip}/32"]
  }

  egress {
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }
}

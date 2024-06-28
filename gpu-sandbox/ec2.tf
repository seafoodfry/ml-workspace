
locals {
  vars = {
    #some_address = aws_instance.some.private_ip
  }
}

module "linux_vanilla" {
  count  = 1
  source = "../modules/ec2s/linux/vanilla"

  name              = "dev"
  ami               = "ami-04064f2a9939d4f29"
  type              = "t3.xlarge"
  security_group_id = aws_security_group.ssh.id
  subnet_id         = module.vpc.public_subnets[0]
  ec2_key_name      = var.ec2_key_name

  instance_profile_name = aws_iam_instance_profile.dcv.name
}
output "linux_vanilla_dns" {
  value       = module.linux_vanilla[*].public_dns
  description = "Public dev DNS"
}

module "linux_gpu" {
  count  = 0
  source = "../modules/ec2s/linux/gpu"

  name                  = "linux-gpu"
  ami                   = "ami-0c4b8684fc96c1de0"
  type                  = "g4dn.xlarge"
  security_group_id     = aws_security_group.ssh.id
  subnet_id             = module.vpc.public_subnets[0]
  ec2_key_name          = var.ec2_key_name
  instance_profile_name = aws_iam_instance_profile.dcv.name
}
output "linux_gpu_dns" {
  value       = module.linux_gpu[*].public_dns
  description = "Public linux GPU DNS"
}

module "windows_gpu" {
  count  = 0
  source = "../modules/ec2s/windows/gpu"

  name = "windows-gpu"
  #ami               = "ami-026433ab26d8782d3"  # GPU
  #type              = "g4dn.xlarge"
  ami               = "ami-0aa76edf764a3a139" # Non-GPU
  type              = "t3.xlarge"
  security_group_id = aws_security_group.rdp.id
  subnet_id         = module.vpc.public_subnets[0]
  ec2_key_name      = var.ec2_key_name
}
output "windows_gpu_dns" {
  value       = module.windows_gpu[*].public_dns
  description = "Public windows GPU DNS"
}
output "windows_gpu_ids" {
  value       = module.windows_gpu[*].instance_id
  description = "Public windows GPU IDs"
}
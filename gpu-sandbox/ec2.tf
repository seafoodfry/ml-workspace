
module "linux_vanilla" {
  count  = 0
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

# The AMI recommended by the launch wizard is
# ami-0862be96e41dcbf74 (64-bit (x86)) / ami-03bfe38a90ce33425 (64-bit (Arm)).
# We got the info about it with:
#   ./run-cmd-in-shell.sh aws ec2 describe-images --image-ids ami-0862be96e41dcbf74
# We got candidates with the command:
# ./run-cmd-in-shell.sh aws ec2 describe-images --owners 099720109477 --filters "Name=platform-details,Values=Linux/UNIX" "Name=architecture,Values=x86_64"  "Name=name,Values=*ubuntu-noble*" "Name=creation-date,Values=2024-08-19*" "Name=description,Values=*Ubuntu*" > out.json
module "ubuntu_metal" {
  count  = 0
  source = "../modules/ec2s/linux/vanilla"

  name              = "firecracker"
  ami               = "ami-0925bd884b1bc0900"
  type              = "c5.metal"
  spot_max_price    = "2.0"
  security_group_id = aws_security_group.ssh.id
  subnet_id         = module.vpc.public_subnets[2]
  ec2_key_name      = var.ec2_key_name

  # Customizations.
  install_docker = true

  instance_profile_name = aws_iam_instance_profile.dcv.name
}
output "ubuntu_metal_dns" {
  value       = module.ubuntu_metal[*].public_dns
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
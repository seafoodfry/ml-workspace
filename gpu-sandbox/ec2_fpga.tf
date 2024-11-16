# ami-0762d104e2c4fb01f is the "FPGA Developer" AMI based on CentOS.
# There is also one based on AL2, but funny enough that one is easier
# to find when going through the marketplace.
# ./run-cmd-in-shell.sh aws ec2 describe-images --image-ids ami-0762d104e2c4fb01f
# ./run-cmd-in-shell.sh aws ec2 describe-images --filters "Name=name,Values=*FPGA*" --region us-east-1
# ./run-cmd-in-shell.sh aws ec2 describe-instance-type-offerings --filters Name=instance-type,Values=f1.2xlarge --location-type availability-zone --region us-east-1
module "fpga" {
  count  = 0
  source = "../modules/ec2s/linux/fpga"

  name = "dev-fpga"
  ami  = "ami-0e178635787eb5e00" # us-east-1.
  #ami               = "ami-0762d104e2c4fb01f" # us-east-2
  type              = "f1.2xlarge"
  security_group_id = aws_security_group.ssh.id
  subnet_id         = module.vpc.public_subnets[0]
  ec2_key_name      = var.ec2_key_name

  instance_profile_name = aws_iam_instance_profile.dcv.name
}
output "fpga" {
  value       = module.fpga[*].public_dns
  description = "Public fpga DNS"
}
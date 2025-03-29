# ./run-cmd-in-shell.sh aws ec2 describe-images --image-ids ami-0851312e7e2e98398
# ./run-cmd-in-shell.sh aws ec2 describe-images --filters "Name=name,Values=*PyTorch 2.6*" "Name=creation-date,Values=2024-11-*" > ml.json
# ./run-cmd-in-shell.sh aws ec2 describe-instance-type-offerings --filters Name=instance-type,Values=g6e.xlarge --location-type availability-zone
#
# For the G5g:
# ./run-cmd-in-shell.sh aws ec2 describe-images --region us-east-1 --filters "Name=description,Values=*G5g*" "Name=creation-date,Values=2025-03-*" "Name=name,Values=*Ubuntu*" > ml2.json
#
# ssh -L 8888:127.0.0.1:8888 ubuntu@${EC2}
#
# tmux new -s jupyter
# source activate pytorch
# jupyter lab --ip=0.0.0.0
# tmux detach or Ctrl+b d
# tmux ls
#
# rsync -rvzP ./llm-book ubuntu@${EC2}:/home/ubuntu
# rsync -rvzP ubuntu@${EC2}:/home/ubuntu/llm-book/ ./llm-book
# scp ubuntu@${EC2}:/home/ubuntu/llm-book/pytorch-101.ipynb ./llm-book
#
# source /opt/pytorch/bin/activate
#
# rsync -rvzP ubuntu@${EC2}:/home/ubuntu/src/ ./vision/cnns --exclude='data'
module "linux_gpu_pytorch" {
  count  = 1
  source = "../modules/ec2s/linux/gpu"

  name                  = "linux-gpu"
  ami                   = "ami-016381122f7637982"
  type                  = "g5g.xlarge"
  security_group_id     = aws_security_group.ssh[0].id
  subnet_id             = module.vpc[0].public_subnets[0]
  ec2_key_name          = var.ec2_key_name
  instance_profile_name = aws_iam_instance_profile.dcv.name
}
output "linux_gpu_pytorch_dns" {
  value       = module.linux_gpu_pytorch[*].public_dns
  description = "Public linux GPU with pytorch DNS"
}
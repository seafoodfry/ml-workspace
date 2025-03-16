# Got AMI from console.
# For details:
# ./run-cmd-in-shell.sh aws ec2 describe-images --region us-east-1 --image-ids ami-0a7a4e87939439934
#
# For the G5g:
# ./run-cmd-in-shell.sh aws ec2 describe-images --region us-east-1 --filters "Name=description,Values=*G5g*" "Name=creation-date,Values=2025-03-*" "Name=name,Values=*Ubuntu*" > ml2.json
#
# Workflow:
#
# ssh -L 8888:127.0.0.1:8888 ubuntu@${EC2}
# WORKDIR=/home/ubuntu/src
# WORKDIR=/mnt/efs/src
# rsync -rvzP gromacs ubuntu@${EC2}:${WORKDIR} --exclude='.venv'
# rsync -rvzP ubuntu@${EC2}:${WORKDIR}/gromacs . --exclude='.venv'
#
# tmux new -s jupyter
# uv run jupyter lab --ip=0.0.0.0
# tmux detach  or Ctrl+b d
# tmux ls
#
# tmux attach -t jupyter
# tmux kill-session -t jupyter
module "gromacs" {
  count  = 0
  source = "../modules/ec2s/linux/gromacs"

  name = "gromacs-${count.index}"
  #ami               = "ami-0a7a4e87939439934"
  #type              = "c8g.2xlarge" #"t4g.xlarge"
  ami               = "ami-081cd0535a43624bf"
  type              = "g5g.xlarge"
  security_group_id = aws_security_group.ssh_and_mpi[0].id
  subnet_id         = module.vpc[0].public_subnets[0]
  ec2_key_name      = var.ec2_key_name
}

output "gromacs_dns" {
  value       = module.gromacs[*].public_dns
  description = "Public dev DNS"
}

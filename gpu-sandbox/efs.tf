# Create EFS file system - One Zone for better performance
resource "aws_efs_file_system" "gromacs" {
  count = length(module.vpc) > 0 ? 1 : 0

  creation_token = "gromacs"

  # Use One Zone for better performance.
  # See https://docs.aws.amazon.com/efs/latest/ug/performance.html
  # NOTE: make sure the index matches the subnet being used by the EC2!
  availability_zone_name = local.selected_azs[0]

  throughput_mode = "elastic"

  lifecycle_policy {
    transition_to_ia = "AFTER_7_DAYS"
  }

  tags = {
    Name = "GromacsEFS-OneZone"
  }
}

resource "aws_security_group" "efs" {
  count = length(module.vpc) > 0 ? 1 : 0

  name        = "gromacs-efs"
  description = "Allow NFS traffic for EFS"
  vpc_id      = module.vpc[0].vpc_id

  ingress {
    description     = "NFS from GROMACS EC2 instances"
    from_port       = 2049
    to_port         = 2049
    protocol        = "tcp"
    security_groups = [aws_security_group.ssh_and_mpi[0].id]
  }
}

# Create mount target in the subnet where your EC2 instances will be.
resource "aws_efs_mount_target" "gromacs_efs_mount" {
  file_system_id  = aws_efs_file_system.gromacs[0].id
  subnet_id       = module.vpc[0].public_subnets[0]
  security_groups = [aws_security_group.efs[0].id]
}


output "efs_dns_name" {
  value = aws_efs_file_system.gromacs[*].dns_name
}

output "gromacs_node_dns" {
  value = module.gromacs[*].private_dns
}
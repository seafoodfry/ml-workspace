output "public_dns" {
  value       = aws_instance.ec2.public_dns
  description = "EC2 public DNS"
}

output "private_dns" {
  value       = aws_instance.ec2.private_dns
  description = "EC2 private DNS"
}
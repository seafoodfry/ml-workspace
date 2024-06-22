output "public_dns" {
  value       = aws_instance.ec2.public_dns
  description = "EC2 Public DNS"
}

output "instance_id" {
  value       = aws_instance.ec2.id
  description = "Instance ID"
}

locals {
  vars = {
    #some_address = aws_instance.some.private_ip
  }
}

resource "aws_instance" "gpu" {
  instance_market_options {
    market_type = "spot"
    spot_options {
      max_price = "0.6" # https://aws.amazon.com/ec2/spot/pricing/
    }
  }

  ami           = "ami-0c4b8684fc96c1de0"
  instance_type = "g4dn.xlarge"

  associate_public_ip_address = true
  vpc_security_group_ids      = [aws_security_group.ssh.id]
  subnet_id                   = module.vpc.public_subnets[0]
  key_name                    = var.ec2_key_name
  root_block_device {
    delete_on_termination = true
    encrypted             = true
    volume_type           = "gp3"
    volume_size           = 135
  }
  metadata_options {
    http_tokens = "required"
  }

  user_data = base64encode(templatefile("${path.module}/setup.sh.tpl", local.vars))

  tags = {
    Name = "gpu"
  }
}

output "ubuntu" {
  value       = aws_instance.gpu.public_dns
  description = "Public DNS"
}

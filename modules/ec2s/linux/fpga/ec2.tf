
locals {
  vars = {
    #install_docker = var.install_docker
  }
}

resource "aws_instance" "ec2" {
  instance_market_options {
    market_type = "spot"
    spot_options {
      max_price = var.spot_max_price
    }
  }

  ami           = var.ami
  instance_type = var.type

  associate_public_ip_address = true
  vpc_security_group_ids      = [var.security_group_id]
  subnet_id                   = var.subnet_id
  key_name                    = var.ec2_key_name

  root_block_device {
    delete_on_termination = true
    encrypted             = true
    volume_type           = "gp3"
    volume_size           = var.volume_size
  }

  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  iam_instance_profile = var.instance_profile_name
  user_data            = base64encode(templatefile("${path.module}/setup.sh.tpl", local.vars))

  tags = {
    Name = var.name
  }
}
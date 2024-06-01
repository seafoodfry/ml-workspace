variable "my_ip" {
  type = string
}

variable "ec2_key_name" {
  type    = string
  default = "numerical-recipes"
}

variable "gpus" {
  type = number
  description = "Number of GPU instances to spin up"
  default = 0
}

variable "dev_machines" {
  type = number
  description = "Number of non-GPU instances to spin up"
  default = 1
}
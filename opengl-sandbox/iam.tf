data "aws_iam_policy_document" "trust_relationship" {
  statement {
    effect = "Allow"
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
    actions = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role" "dcv" {
  name               = "dcv-opengl"
  path               = "/"
  assume_role_policy = data.aws_iam_policy_document.trust_relationship.json
}

resource "aws_iam_instance_profile" "dcv" {
  name = "dcv"
  role = aws_iam_role.dcv.name
}

data "aws_iam_policy_document" "dcv" {
  statement {
    actions = ["s3:GetObject"]
    resources = ["arn:aws:s3:::dcv-license.${data.aws_region.current.name}/*"]
    effect    = "Allow"
  }
}

resource "aws_iam_policy" "s3_get_policy" {
  name        = "dcv-opengl"
  description = "https://docs.aws.amazon.com/dcv/latest/adminguide/setting-up-license.html"
  policy      = data.aws_iam_policy_document.dcv.json
}


resource "aws_iam_role_policy_attachment" "dcv" {
  role       = aws_iam_role.dcv.name
  policy_arn = aws_iam_policy.s3_get_policy.arn
}
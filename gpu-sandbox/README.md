# GPU Sandbox

We assume you have followed the steps in [https://seafoodfry.github.io aws-lab-setup](https://seafoodfry.github.io//aws/lab/2024/05/27/aws-lab-setup/).


We recommend you use [github.com/tfutils/tfenv](https://github.com/tfutils/tfenv) to manage the Terraform versions you will need.

For this example we did
```
tfenv init
tfenv install 1.8.4
tfenv use 1.8.4
```

You'll also need

```
brew install jq
```

---


## Spining Up a GPU

### Finding an AMI

While checking out what AMIs were recommended through the launch wizard, we came across the
AMI ID `ami-0296a329aeec73707` published by amazon with the title
"Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.2.0 (Amazon Linux 2) 20240521".
We can query info about it as follows:

```
./run-cmd-in-shell.sh aws ec2 describe-images --owners amazon --image-ids ami-0296a329aeec73707
```

We kept searching for AMIs with the following query

```
./run-cmd-in-shell.sh aws ec2 describe-images --owner 898082745236 --filters "Name=platform-details,Values=Linux/UNIX" "Name=architecture,Values=x86_64"  "Name=name,Values=*Amazon Linux 2*" "Name=creation-date,Values=2024-05*" "Name=description,Values=*G4dn*" > out.json
```

and found this candidate

```json
{
    "Architecture": "x86_64",
    "CreationDate": "2024-05-22T09:42:47.000Z",
    "ImageId": "ami-0c4b8684fc96c1de0",
    "ImageLocation": "amazon/Deep Learning OSS Nvidia Driver AMI (Amazon Linux 2) Version 78.2",
    "ImageType": "machine",
    "Public": true,
    "OwnerId": "898082745236",
    "PlatformDetails": "Linux/UNIX",
    "UsageOperation": "RunInstances",
    "State": "available",
    "BlockDeviceMappings": [
        {
            "DeviceName": "/dev/xvda",
            "Ebs": {
                "DeleteOnTermination": true,
                "Iops": 3000,
                "SnapshotId": "snap-0af15a9e4c4b2e59c",
                "VolumeSize": 105,
                "VolumeType": "gp3",
                "Throughput": 125,
                "Encrypted": false
            }
        }
    ],
    "Description": "Supported EC2 instances: G4dn, G5, G6, Gr6, P4d, P4de, P5. PyTorch-2.1, TensorFlow-2.16. Release notes: https://docs.aws.amazon.com/dlami/latest/devguide/appendix-ami-release-notes.html",
    "EnaSupport": true,
    "Hypervisor": "xen",
    "ImageOwnerAlias": "amazon",
    "Name": "Deep Learning OSS Nvidia Driver AMI (Amazon Linux 2) Version 78.2",
    "RootDeviceName": "/dev/xvda",
    "RootDeviceType": "ebs",
    "SriovNetSupport": "simple",
    "VirtualizationType": "hvm",
    "DeprecationTime": "2026-05-22T09:42:47.000Z"
},
```

### Running the GPU


Run the followi
```
terraform init
```

We will set the `my_ip` variable as follows
```
export TF_VAR_my_ip=$(curl https://cloudflare.com/cdn-cgi/trace | grep ip | awk -F= '{print $2}')
```
(We could have also set it using `-var my_ip="x.x.x.x"`.)

Then,
```
./run-cmd-in-shell.sh terraform plan -out a.plan
```

Then apply the plan
```
./run-cmd-in-shell.sh terraform apply a.plan
```

To clean up
```
./run-cmd-in-shell.sh terraform destroy
```

You can find logs in
```
cat /var/log/cloud-init-output.log
```

---

## SSH Tips

To copy files
```
scp Makefile ubuntu@${EC2}:/home/ubuntu
```

Or
```
scp ubuntu@${EC2}:/home/ubuntu/Animations-101.ipynb .
```

Note that for Amazon Linux 2 EC2s the default username is `ec2-user`.

---

## Jupyter Tips

There are ready to use images in
[quay.io/organization/jupyter](https://quay.io/organization/jupyter).
That repository is documented in
[jupyter-docker-stacks](https://jupyter-docker-stacks.readthedocs.io/en/latest/).

```
docker run -p 8888:8888 quay.io/jupyter/scipy-notebook:python-3.11
```

You can then view it locally by running
```
ssh -L 8888:127.0.0.1:8888 ubuntu@${EC2}
```


---

## Cuda by Example

This sandbox was created to follow along with
[developer.nvidia.com/cuda-example](https://developer.nvidia.com/cuda-example).
We downloaded the zip file and its contents are in [cuda_by_example](./cuda_by_example).

Since we did this on a mac, we also checked if the aurantine attribute was associated with the files
with
```
ls -l@
```

And we removed it the following command
```
xattr -r -d com.apple.quarantine cuda_by_example
```
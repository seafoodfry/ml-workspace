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

### Finding a GPU AMI

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


### Finding a Non-GPU AMI

Following the procedure outlined below, we saw the AMI `ami-0ca2e925753ca2fb4` as one of the recommende
AMIs when trying to launch an EC2 from the console.
Running

```
./run-cmd-in-shell.sh aws ec2 describe-images --image-ids ami-0ca2e925753ca2fb4
```
Gave us
```json
{
    "Images": [
        {
            "Architecture": "x86_64",
            "CreationDate": "2024-05-24T03:27:51.000Z",
            "ImageId": "ami-0ca2e925753ca2fb4",
            "ImageLocation": "amazon/al2023-ami-2023.4.20240528.0-kernel-6.1-x86_64",
            "OwnerId": "137112412989",
            "PlatformDetails": "Linux/UNIX",
            "Description": "Amazon Linux 2023 AMI 2023.4.20240528.0 x86_64 HVM kernel-6.1",
            "ImageOwnerAlias": "amazon",
            "Name": "al2023-ami-2023.4.20240528.0-kernel-6.1-x86_64",
            "DeprecationTime": "2024-08-22T03:28:00.000Z"
            ...
        }
    ]
}
```

So we looked for the newest version with
```
./run-cmd-in-shell.sh aws ec2 describe-images --owner amazon --filters "Name=platform-details,Values=Linux/UNIX" "Name=architecture,Values=x86_64" "Name=creation-date,Values=2024-05*" "Name=description,Values=*Amazon Linux*" --query 'Images[?!contains(Description, `ECS`) && !contains(Description, `EKS`) && !contains(Description, `gp2`)]' > out.json
```

We ended up going with
```json
{
        "Architecture": "x86_64",
        "CreationDate": "2024-05-30T00:51:59.000Z",
        "ImageId": "ami-04064f2a9939d4f29",
        "ImageLocation": "amazon/amzn2-ami-kernel-5.10-hvm-2.0.20240529.0-x86_64-ebs",
        "ImageType": "machine",
        "Public": true,
        "OwnerId": "137112412989",
        "PlatformDetails": "Linux/UNIX",
        "UsageOperation": "RunInstances",
        "State": "available",
        "BlockDeviceMappings": [
            {
                "DeviceName": "/dev/xvda",
                "Ebs": {
                    "DeleteOnTermination": true,
                    "SnapshotId": "snap-07f3b72092a551eb6",
                    "VolumeSize": 8,
                    "VolumeType": "standard",
                    "Encrypted": false
                }
            }
        ],
        "Description": "Amazon Linux 2 Kernel 5.10 AMI 2.0.20240529.0 x86_64 HVM ebs",
        "EnaSupport": true,
        "Hypervisor": "xen",
        "ImageOwnerAlias": "amazon",
        "Name": "amzn2-ami-kernel-5.10-hvm-2.0.20240529.0-x86_64-ebs",
        "RootDeviceName": "/dev/xvda",
        "RootDeviceType": "ebs",
        "SriovNetSupport": "simple",
        "VirtualizationType": "hvm",
        "DeprecationTime": "2025-07-01T00:00:00.000Z"
    },
```
Note that we chose an Amazon Linux 2 AMI for compatibility with our GPU instances.


### Running the GPU


Run the following
```
./run-cmd-in-shell.sh terraform init
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


---

## Testing the GPU

Try out the compiler
```
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Mon_Apr__3_17:16:06_PDT_2023
Cuda compilation tools, release 12.1, V12.1.105
Build cuda_12.1.r12.1/compiler.32688072_0
```

Try the system management interface
```
$ nvidia-smi
Wed May 29 02:48:55 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.161.08             Driver Version: 535.161.08   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       On  | 00000000:00:1E.0 Off |                    0 |
| N/A   24C    P8               8W /  70W |      2MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```


Now we can actually try some code.
Copy the source,
```
scp -r our-cuda-by-example ec2-user@${EC2}:/home/ec2-user/src
```

```
dcv list-endpoints -j
```
# GPU Sandbox

The following outlines how we configured our AWS account
[seafoodfry.github.io: aws-lab-setup](https://seafoodfry.github.io//aws/lab/2024/05/27/aws-lab-setup/).

The following outlines the basics of our TF workspace
[seafoodfry.github.io: graphics-pt-01](https://seafoodfry.github.io//aws/lab/gpu/graphics/2024/06/21/graphics-pt-01/).


---

### Finding a Windows GPU AMI

We found the following AMIs using the EC2 launch wizard: turns out that the secret is to search the marketplace, search for whatever you want, click select, and then "subscribe on instance launch".
Once you do that the AMIs will show themselves.

NICE DCV for Windows (g4 and g5 with NVIDIA gaming driver)
```
DCV-Windows-2023.1.16388-NVIDIA-gaming-555.99-2024-06-14T21-47-06.900Z
ami-0871751821043a991
```


Microsoft Windows Server 2019 with NVIDIA GRID Driver
```
Windows_Server-2019-English-Full-G3-2024.06.13-b6156132-9d29-4061-9e84-fc9f8357376c
ami-09d27e664fc45deff
```


Microsoft Windows Server 2019 with NVIDIA Tesla Driver
```
Windows_Server-2019-English-Tesla-2024.06.13
ami-026433ab26d8782d3
```


For non-GPU AMIs we did

```
./run-cmd-in-shell.sh aws ec2 describe-images --owner amazon --filters "Name=platform,Values=windows" "Name=architecture,Values=x86_64" "Name=creation-date,Values=2024-06*" "Name=description,Values=*Windows Server*" "Name=name,Values=*English*" --query 'Images[?!contains(Description, `"2016"`) && !contains(Description, `SQL`) && !contains(Description, `EKS`) && !contains(Description, `ECS`) ]' > out.json
```


### Running the EC2


Commands to get things up and running
```
./run-cmd-in-shell.sh terraform init

export TF_VAR_my_ip=$(curl https://cloudflare.com/cdn-cgi/trace | grep ip | awk -F= '{print $2}')

./run-cmd-in-shell.sh terraform plan -out a.plan

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


But even easier is to do
```
rsync -rvzP our-cuda-by-example ec2-user@${EC2}:/home/ec2-user/src
```
and
```
rsync -rvzP ec2-user@${EC2}:/home/ec2-user/src/our-cuda-by-example/ our-cuda-by-example
```

For the opengl code

```
rsync -rvzP learning-opengl ec2-user@${EC2}:/home/ec2-user/src
```
and
```
rsync -rvzP ec2-user@${EC2}:/home/ec2-user/src/learning-opengl/ learning-opengl
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

## Mac Tips

Checked if the aurantine attribute was associated with the files with
```
ls -l@
```

And we removed it the following command
```
xattr -r -d com.apple.quarantine cuda_by_example
```



---

## Windows Tips

The log file for EC2Launch is `C:\ProgramData\Amazon\EC2-Windows\Launch\Log\UserdataExecution.log`.
The log file for EC2Launch v2 is `C:\ProgramData\Amazon\EC2Launch\log\agent.log`.

[How Amazon EC2 handles user data for Windows instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/user-data.html#ec2-windows-user-data)
outlines how you may go about accessing these log files.

We actually found the userdata execution log in
`C:\ProgramData\Amazon\EC2-Windows\Launch\Log\UserdataExecution.log`.



---

## Visual Studio

To enable SSE2
1. Project Properties
2. Configuration Properties
3. C/C++
4. Code Generation
5. Enable Enhanced Instruction Set


To enable fast floating point model `/fp:fast` for all platforms
1. Project Properties
2. Configuration Properties
3. C/C++
4. Code Generation
5. Floating Point Model
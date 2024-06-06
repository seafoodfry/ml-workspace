# OpenGL Sandbox

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

See the README in the GPU sandbox directory.


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

## SSH Cheatsheet


Now we can actually try some code.

First, copy the source,
```
scp -r our-cuda-by-example ec2-user@${EC2}:/home/ec2-user/src
```

There are two good sources of docs here
1. [What is DCV](https://docs.aws.amazon.com/dcv/latest/adminguide/what-is-dcv.html). You'll really need to read the docs though!
2. [Deploy an EC2 instance with NICE DCV](https://www.hpcworkshops.com/06-nice-dcv/standalone/08-deploy-ec2.html)

Once the EC2 is ready we will perform the following checks.
First check, taken from 
[Prerequisites for Linux NICE DCV servers](https://docs.aws.amazon.com/dcv/latest/adminguide/setting-up-installing-linux-prereq.html#linux-prereq-xserver)
```
sudo DISPLAY=:0 XAUTHORITY=$(ps aux | grep "X.*\-auth" | grep -v grep | sed -n 's/.*-auth \([^ ]\+\).*/\1/p') glxinfo | grep -i "opengl.*version"
```

Then we will perform a couple more commands from
[Post-Installation checks](https://docs.aws.amazon.com/dcv/latest/adminguide/setting-up-installing-linux-checks.html)
```
sudo DISPLAY=:0 XAUTHORITY=$(ps aux | grep "X.*\-auth" | grep -v grep | sed -n 's/.*-auth \([^ ]\+\).*/\1/p') xhost | grep "SI:localuser:dcv$"
```
This one is ok if it doesn't return anything.

```
sudo DISPLAY=:0 XAUTHORITY=$(ps aux | grep "X.*\-auth" | grep -v grep | sed -n 's/.*-auth \([^ ]\+\).*/\1/p') xhost | grep "LOCAL:$"
```
This one should return something.

This one should return no errors, maybe just an info item.
```
sudo dcvgldiag
```

To check that the DCV server is running do
```
sudo systemctl status dcvserver
```

And to get the fingerprint of its self-signed certificate (we'll needed when we actually sign in)
```
dcv list-endpoints -j
```

Now, we need to give `ec2-user` an actual password
```
sudo passwd ec2-user
```

And create a session.
```
dcv create-session dcvdemo
```

At thsi point we are ready to use NICE DCV.


[github.com/Dav1dde/glad](https://github.com/Dav1dde/glad)

1. Language: C/C++
2. API: gl Version 4.6
3. Profile: Core
4. Options: check the box for "Generate a loader"
5. Click Generate
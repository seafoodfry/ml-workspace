
all:
	docker run \
		-v $(PWD):/home/jovyan \
		-p 8888:8888 \
		quay.io/jupyter/scipy-notebook:python-3.11

clean:
	rm -f a.plan
	rm -f terraform.tfstate*
	rm -rf .terraform*

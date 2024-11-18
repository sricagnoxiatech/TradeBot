NAMESPACE=hypertrade
GITHUB_USERNAME=sricagnoxiatech

dev:
	scripts/prepare.sh development
	skaffold dev --profile=development --tail --default-repo=ghcr.io/$(GITHUB_USERNAME)

stop:
	minikube stop

build:
	scripts/prepare.sh production
	skaffold build --profile production --default-repo=ghcr.io/$(GITHUB_USERNAME)

prod:
	scripts/prepare.sh production
	skaffold run --profile production --default-repo=ghcr.io/$(GITHUB_USERNAME)

	connect:
	doctl kubernetes cluster kubeconfig save $(NAMESPACE)-cluster
	kubectl port-forward svc/proxy 8080:8080 --namespace=$(NAMESPACE)

disconnect:
	doctl kubernetes cluster kubeconfig remove $(NAMESPACE)-cluster

delete:
	kubectl delete --all deployments
	minikube delete
from metrics import bleu

generated_example = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: hello-backend
  labels:
    app: hello
    tier: backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hello
      tier: backend
  template:
    metadata:
      labels:
        app: hello
        tier: backend
        track: stable
    spec:
      containers:
      - name: hello-backend
        image: gcr.io/google-samples/hello-backend:v2
        ports:
        - containerPort: 80"""
reference_example = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
spec:
  selector:
    matchLabels:
      app: hello
      tier: backend
      track: stable
  replicas: 3
  template:
    metadata:
      labels:
        app: hello
        tier: backend
        track: stable
    spec:
      containers:
        - name: hello
          image: "gcr.io/google-samples/hello-go-gke:1.0"
          ports:
            - name: http
              containerPort: 80"""

if __name__ == "__main__":
    bleu_score = bleu.test(generated_example, reference_example)
    print(f"BLEU score: {bleu_score:.4f}")
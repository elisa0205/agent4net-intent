from metrics import bleu, codeBleu, edit_distance, exact_match, kv_match, kv_wildcard, label_match

generated_example = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: cpu-demo
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cpu-demo
  template:
    metadata:
      labels:
        app: cpu-demo
    spec:
      containers:
      - name: cpu-demo
        image: gcr.io/google-samples/hpa-example
        ports:
        - containerPort: 80
        resources:
          requests:
            cpu: 200m
          limits:
            cpu: 500m
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: cpu-demo-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cpu-demo
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 30
---
apiVersion: v1
kind: Service
metadata:
  name: cpu-demo-svc
spec:
  selector:
    app: cpu-demo
  ports:
  - port: 8080
    targetPort: 80
  type: ClusterIP
"""
reference_example = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: cpu-demo
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cpu-demo
  template:
    metadata:
      labels:
        app: cpu-demo
    spec:
      containers:
      - name: cpu-demo
        image: nginx:latest
        resources:
          requests:
            cpu: 200m
          limits:
            cpu: 500m
---
apiVersion: v1
kind: Service
metadata:
  name: cpu-demo-svc
spec:
  selector:
    app: cpu-demo
  ports:
    - port: 80
      targetPort: 8080
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: cpu-demo-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cpu-demo
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 30"""

'''
RESULTS:
  BLEU score: 0.8635
  Edit Distance score: 0.6222
  Exact Match score: False
  Label Match score: 0.9655
  KV Match score: 0.6552
  KV Wildcard score: 0.8966
'''

if __name__ == "__main__":

    bleu_score = bleu.test(generated_example, reference_example)
    print(f"BLEU score: {bleu_score:.4f}")

    code_bleu_score = codeBleu.test(generated_example, reference_example)
    print(f"CodeBLEU score: {code_bleu_score:.4f}")
 
    edit_distance_score = edit_distance.test(generated_example, reference_example)
    print(f"Edit Distance score: {edit_distance_score:.4f}")

    exact_match_score = exact_match.test(generated_example, reference_example)
    print(f"Exact Match score: {exact_match_score}")

    label_match_score = label_match.test(generated_example, reference_example)
    print(f"Label Match score: {label_match_score:.4f}")

    kv_match_score = kv_match.test(generated_example, reference_example)
    print(f"KV Match score: {kv_match_score:.4f}")

    kv_wildcard_score = kv_wildcard.test(generated_example, reference_example)
    print(f"KV Wildcard score: {kv_wildcard_score:.4f}")



    



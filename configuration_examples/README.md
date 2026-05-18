# Configuration examples
This folder contains a collection of Kubernetes configuration examples demonstrating different architectural patterns and cluster objects.
Each subfolder represents a distinct scenario, with a specific focus on resources, networking, scalability, persistence, or environment isolation.

## Overview

### stateless-app
Minimal example of a stateless application.
Shows gthe basic case of an application using a Deployment and a Service.

### stateful-app
Example of a simpler stateful application.
Shows how to persist database data using a PersistenVolume object.

### secure-stateful-app
Example of a secure stateful application.
Shows data persistence, stable pod identities, and secure credential management using a Secret object.

### prod-dev-example
Example of separation between development and production environments.
Demonstrates logical isolation between different environments and different namespaces.

### php-guestbook-example
Guestbook example with PHP frontend and Redis backend.
Shows a typical web architecture with Redis backend, separating read and write operations between leader and replicas.

### HPA-example
Example dedicated to horizontal pod autoscaling.
Demonstrates how to automatically scale pods based on resource utilization, particularly CPU.

### fronend-backend-app
Example of a multi-tier application with frontend and backend separated.
A LoadBalancer expose the frontend externally and a NetworkPolicy object allow the access to the backand tier only by pods in the frontend tier.

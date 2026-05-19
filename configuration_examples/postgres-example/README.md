# postgres-example

Example of a small multi-component web application composed of a frontend, an internal API, and a PostgreSQL database.

This folder demonstrates:
- a frontend exposed through an Ingress
- a backend API service kept internal
- a PostgreSQL StatefulSet with persistent storage
- a ConfigMap for application and Nginx settings
- a Secret for database credentials
- NetworkPolicies that isolate the API and database traffic

Suggested apply order:
1. namespace
2. ConfigMap and Secret
3. PostgreSQL service and StatefulSet
4. API service and deployment
5. frontend service and deployment
6. NetworkPolicy
7. Ingress

Replace the sample images with your own application images if you want the API to execute real business logic.
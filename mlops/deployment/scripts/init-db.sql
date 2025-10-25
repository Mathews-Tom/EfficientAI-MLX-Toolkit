-- Initialize databases for MLOps services
-- This script runs on PostgreSQL container startup

-- Create MLFlow database
CREATE DATABASE mlflow;

-- Create Airflow database
CREATE DATABASE airflow;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlops;
GRANT ALL PRIVILEGES ON DATABASE airflow TO mlops;

-- Create extensions
\c mlflow;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

\c airflow;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

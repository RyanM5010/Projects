# Distributed System: Skier Lift Ride

This project implements a scalable distributed backend system to handle skier lift ride events efficiently and reliably.

## Overview

The architecture follows a distributed pattern, using message queues for decoupled communication, Redis for caching, and is deployed on AWS.

📄 [Click here to view the system architecture (PDF)](./Overview.pdf)

## Features

- 🚡 Handles 4K+ lift ride events per second
- 📬 RabbitMQ-based asynchronous messaging
- 🧠 Redis caching for fast reads
- ☁️ Deployed using AWS EC2 and Load Balancers
- 📊 CloudWatch and custom metrics for observability

## Tech Stack

- **Language**: Java
- **Framework**: Spring Boot
- **Message Queue**: RabbitMQ
- **Database**: DynamoDB
- **Cache**: Redis
- **Deployment**: AWS (EC2, S3, Load Balancer)

## Getting Started

```bash
cd backend-spring
./mvnw spring-boot:run

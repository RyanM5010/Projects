# Distributed System: Skier Lift Ride Tracking

This project implements a scalable, distributed backend system to handle large volumes of skier lift ride data using multithreaded clients, RESTful APIs, and messaging queues.

## System Overview

<img src="./images/overview.png" alt="Overview Diagram" width="600"/>

## Components

- **Client**: Multithreaded generator sending POST requests
- **Server**: RESTful APIs built with Spring Boot
- **Consumer**: Message queue processor (RabbitMQ or Redis)
- **JMeter**: Load testing and benchmarking

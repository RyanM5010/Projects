# CS6650 Skier Client - Assignment 1

## Setup and Running Instructions

This guide explains how to set up and run the client for **CS6650 Skier Assignment 1**.

---

## 1. Install `java-client-generated` for Swagger API
Before running the client, install the generated API client dependencies:

```sh
cd cs6650-skier/Assignment1/java-client-generated
mvn clean install
```

> **Note:** The `java-client-generated` has been modified to include a dependency for `javax.annotation-api` in `pom.xml`, but no additional changes are required from your end.

---

## 2. Rebuild the Client
Once the API client is installed, rebuild the client project:

```sh
cd cs6650-skier/Assignment1/Client
mvn clean package
```

---

## 3. Configure and Run the Client
### (A) Update the Server URL
Modify the **base path's IP address** in `ClientApp.java` (for both `Part1` and `Part2`) before running the client:

- **File Location:**
  - `Part1/ClientApp.java`
  - `Part2/ClientApp.java`

- Update the line:
  ```java
  private static final String BASE_PATH = "http://<your-EC2-IP>:8080/Server_war";
  ```
  Replace `<your-EC2-IP>` with your actual EC2 instance IP.

---

### (B) Run the Client
1. **Run Part 1 (Basic Client Implementation)**
   - Right-click `ClientApp.java` inside the `Part1` package and select **Run**.
  
2. **Run Part 2 (Client with Performance Metrics)**
   - Right-click `ClientApp.java` inside the `Part2` package and select **Run**.

Alternatively, you can use Maven commands:

```sh
mvn clean install
```

---

## Additional Notes
- The `maven.compiler.source` and `maven.compiler.target` are set to **Java 11** instead of detecting via `java -version`.
- The `javax.annotation-api` dependency was added in `java-client-generated` for compatibility.

---




import io.swagger.client.api.SkiersApi;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.Collections;
import java.util.ArrayList;

public class ClientApp {
  private static final int TOTAL_REQUESTS = 200000;
  private static final int INITIAL_THREADS = 32;
  private static final int REQUESTS_PER_THREAD = 1000;
  private static final int QUEUE_SIZE = 100000;
  private static final String BASE_PATH = "http://35.89.18.167:8080";
  private static final String CSV_FILE = "output3.csv";

  public static void main(String[] args) {
    // ** Check for user input **
    if (args.length < 1) {
      System.err.println("Usage: java ClientApp <dayId>");
      System.exit(1);
    }
    int dayId = Integer.parseInt(args[0]);

    System.out.println("Starting Ski Resort Client for Day " + dayId + "...");

    long startTime = System.currentTimeMillis();

    BlockingQueue<LiftRideEvent> eventQueue = new LinkedBlockingQueue<>(QUEUE_SIZE);
    AtomicInteger successCount = new AtomicInteger(0);
    AtomicInteger failureCount = new AtomicInteger(0);
    AtomicInteger completedRequests = new AtomicInteger(0);
    List<RequestMetric> metrics = Collections.synchronizedList(new ArrayList<>(TOTAL_REQUESTS));

    // ** Update generator with dayId **:
    EventGenerator generator = new EventGenerator(eventQueue, TOTAL_REQUESTS, dayId);
    new Thread(generator).start();

    SkiersApi skiersApi = new SkiersApi();
    skiersApi.getApiClient().setBasePath(BASE_PATH);

    ExecutorService executor = Executors.newCachedThreadPool();
    CountDownLatch firstBatchLatch = new CountDownLatch(INITIAL_THREADS);

    for (int i = 0; i < INITIAL_THREADS; i++) {
      executor.execute(new RequestSender(
          eventQueue, skiersApi, REQUESTS_PER_THREAD, firstBatchLatch,
          successCount, failureCount, completedRequests, metrics
      ));
    }

    try {
      firstBatchLatch.await();
      System.out.println("First batch completed, launching remaining threads...");
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }


    int remainingRequests = TOTAL_REQUESTS - completedRequests.get();
    int additionalThreads = (int) Math.ceil(remainingRequests / (double) REQUESTS_PER_THREAD);
    CountDownLatch allRequestsLatch = new CountDownLatch(additionalThreads);

    for (int i = 0; i < additionalThreads; i++) {
      int threadRequests = Math.min(REQUESTS_PER_THREAD, remainingRequests - (i * REQUESTS_PER_THREAD));

      executor.execute(new RequestSender(
          eventQueue, skiersApi, threadRequests, allRequestsLatch,
          successCount, failureCount, completedRequests, metrics
      ));
    }

    try {
      allRequestsLatch.await();
      executor.shutdown();
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      executor.shutdownNow();
    }

    long endTime = System.currentTimeMillis();
    double wallTime = (endTime - startTime) / 1000.0;
    double throughput = TOTAL_REQUESTS / wallTime;

    System.out.println("\n====== Results ======");
    System.out.println("Successful requests: " + successCount.get());
    System.out.println("Failed requests: " + failureCount.get());
    System.out.println("Wall time: " + wallTime + " seconds");
    System.out.println("Throughput: " + throughput + " requests/second");

    printMetrics(metrics);

    // Write metrics to CSV
    try (FileWriter writer = new FileWriter(CSV_FILE)) {
      writer.write("start_time,request_type,latency,response_code\n");
      for (RequestMetric metric : metrics) {
        writer.write(metric.toCsvString() + "\n");
      }
    } catch (IOException e) {
      System.err.println("Error writing to CSV: " + e.getMessage());
    }

    System.out.println("\n====== Client Configuration Summary ======");
    System.out.println("Total Requests: " + TOTAL_REQUESTS);
    System.out.println("Initial Threads: " + INITIAL_THREADS);
    System.out.println("Final Threads Used: " + additionalThreads);
  }


  private static void printMetrics(List<RequestMetric> metrics) {
    List<Long> latencies = metrics.stream()
        .filter(m -> m.getResponseCode() == 201)
        .map(RequestMetric::getLatency)
        .sorted()
        .collect(Collectors.toList());

    if (latencies.isEmpty()) {
      System.out.println("No successful requests to calculate metrics");
      return;
    }

    double mean = latencies.stream().mapToLong(Long::longValue).average().orElse(0.0);
    long median = latencies.get(latencies.size() / 2);
    long p99 = latencies.get((int) (latencies.size() * 0.99));
    long min = Collections.min(latencies);
    long max = Collections.max(latencies);

    System.out.println("\n====== Performance Metrics ======");
    System.out.println("Mean response time: " + mean + " ms");
    System.out.println("Median response time: " + median + " ms");
    System.out.println("99th percentile response time: " + p99 + " ms");
    System.out.println("Min response time: " + min + " ms");
    System.out.println("Max response time: " + max + " ms");

  }
}

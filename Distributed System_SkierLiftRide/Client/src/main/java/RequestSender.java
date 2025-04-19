import io.swagger.client.ApiException;
import io.swagger.client.api.SkiersApi;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import io.swagger.client.ApiResponse;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.*;

public class RequestSender implements Runnable {

  private final BlockingQueue<LiftRideEvent> eventQueue;
  private final SkiersApi skiersApi;
  private final int numRequests;
  private final CountDownLatch completed;
  private final AtomicInteger successCount;
  private final AtomicInteger failureCount;
  private final AtomicInteger completedRequests;
  private final List<RequestMetric> metrics;
  private static final int MAX_RETRIES = 5;
  private final List<RequestMetric> localMetrics;
  private static final int BATCH_SIZE = 20;

  public RequestSender(BlockingQueue<LiftRideEvent> eventQueue,
      SkiersApi skiersApi,
      int numRequests,
      CountDownLatch completed,
      AtomicInteger successCount,
      AtomicInteger failureCount,
      AtomicInteger completedRequests,
      List<RequestMetric> metrics) {
    this.eventQueue = eventQueue;
    this.skiersApi = skiersApi;
    this.numRequests = numRequests;
    this.completed = completed;
    this.successCount = successCount;
    this.failureCount = failureCount;
    this.completedRequests = completedRequests;
    this.metrics = metrics;
    this.localMetrics = new ArrayList<>(numRequests);
  }

  @Override
  public void run() {
    int sentRequests = 0;
    System.out.println(Thread.currentThread().getName() + " started processing events...");
    
    // Create a batch buffer
    List<LiftRideEvent> batch = new ArrayList<>(BATCH_SIZE);
    List<Long> startTimes = new ArrayList<>(BATCH_SIZE);

    while (sentRequests < numRequests) {
      try {
        // Fetch a batch of events to reduce blocking overhead
        batch.clear();
        startTimes.clear();
        
        for (int i = 0; i < BATCH_SIZE && sentRequests + batch.size() < numRequests; i++) {
          LiftRideEvent event = eventQueue.poll(100, TimeUnit.MILLISECONDS); // Wait up to 100ms
          if (event != null) {
            batch.add(event);
            startTimes.add(System.currentTimeMillis());
          }
        }

        if (batch.isEmpty()) {
          continue;
        }

        // Process the batch
        for (int i = 0; i < batch.size(); i++) {
          LiftRideEvent event = batch.get(i);
          long startTime = startTimes.get(i);
          
          boolean success = sendWithRetry(event);
          long endTime = System.currentTimeMillis();
          long latency = endTime - startTime;
          
          localMetrics.add(new RequestMetric(startTime, "POST", latency, success ? 201 : 500));
          
          if (success) {
            successCount.incrementAndGet();
          } else {
            failureCount.incrementAndGet();
          }
          
          sentRequests++;
          completedRequests.incrementAndGet();
        }

      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        break;
      }
    }

    synchronized (metrics) {
      metrics.addAll(localMetrics);
    }

    completed.countDown();
  }

  private boolean sendWithRetry(LiftRideEvent event) {
    int attempts = 0;
    while (attempts < MAX_RETRIES) {
        try {
            ApiResponse<Void> response = skiersApi.writeNewLiftRideWithHttpInfo(
                event.getLiftRide(), event.getResortID(), event.getSeasonID(), event.getDayID(), event.getSkierID()
            );

            if (response.getStatusCode() == 201) {
                return true; // Successful request
            }
        } catch (ApiException e) {
            attempts++;

            // If last attempt, requeue event instead of sleeping
            if (attempts == MAX_RETRIES) {
                eventQueue.offer(event);  // Put back into queue for retry
                return false;
            }
      }
    }
    return false;

  }
}


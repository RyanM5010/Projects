package consumer;

import software.amazon.awssdk.services.dynamodb.DynamoDbClient;
import software.amazon.awssdk.services.dynamodb.model.*;
import software.amazon.awssdk.regions.Region;

import java.util.Map;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;


public class LiftRideCountsDao {
  private static final String TABLE_NAME = "LiftRideCounts";
  private static final LiftRideCountsDao instance = new LiftRideCountsDao();
  private final DynamoDbClient dynamoDbClient;
  // In-memory buffer to batch counts
  private final ConcurrentHashMap<String, AtomicInteger> counterBuffer = new ConcurrentHashMap<>();
  private final ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();


  private LiftRideCountsDao() {
    this.dynamoDbClient = DynamoDbClient.builder()
        .region(Region.US_WEST_2)  // or your actual region
        .build();

    // Flush buffer every second
    scheduler.scheduleAtFixedRate(this::flushBuffer, 1, 1, TimeUnit.SECONDS);

  }

  // Make singleton
  public static LiftRideCountsDao getInstance() {
    return instance;
  }

  /**
   * Increment counter in memory (non-blocking)
   */
  public void incrementBuffered(String resortSeasonDayId) {
    counterBuffer.computeIfAbsent(resortSeasonDayId, k -> new AtomicInteger(0)).incrementAndGet();
  }

  /**
   * Flushes all pending counts to DynamoDB
   */
  private void flushBuffer() {
    for (Map.Entry<String, AtomicInteger> entry : counterBuffer.entrySet()) {
      String key = entry.getKey();
      int count = entry.getValue().getAndSet(0);
      if (count > 0) {
        updateCount(key, count);
      }
    }
  }

  public void updateCount(String resortSeasonDayID,  int count) {
    try {
      UpdateItemRequest updateRequest = UpdateItemRequest.builder()
          .tableName("LiftRideCounts")
          .key(Map.of("ResortSeasonDayID", AttributeValue.builder().s(resortSeasonDayID).build()))
          .updateExpression("ADD #c :incr")
          .expressionAttributeNames(Map.of("#c", "count"))
          .expressionAttributeValues(Map.of(":incr", AttributeValue.builder().n(String.valueOf(count)).build()))
          .build();

      dynamoDbClient.updateItem(updateRequest);
    } catch (DynamoDbException e) {
      System.err.println("Failed to update counter: " + e.getMessage());
    }
  }

}

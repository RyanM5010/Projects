package consumer;

import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.dynamodb.DynamoDbClient;
import software.amazon.awssdk.services.dynamodb.model.*;

import java.util.*;
import java.util.concurrent.*;

public class SkierLiftRidesDao {
  private final DynamoDbClient dynamoDbClient;
  private final static String TABLE_NAME = "SkierLiftRides";
  private final static int BATCH_SIZE = 25;
  private static final SkierLiftRidesDao instance = new SkierLiftRidesDao();

  private final List<WriteRequest> writeRequests = new ArrayList<>();
  // Periodically flush pending write requests every 1 second
  private final ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();

  public SkierLiftRidesDao() {
    this.dynamoDbClient = DynamoDbClient.builder().region(Region.US_WEST_2).build();
    scheduler.scheduleAtFixedRate(this::flushRemainingWrites, 1, 1, TimeUnit.SECONDS);
  }

  // Make LiftRides Singleton
  public static SkierLiftRidesDao getInstance() {
    return instance;
  }

  // Add a write request to the buffer, flush immediately if batch size reached
  public synchronized void addWriteRequest(WriteRequest writeRequest) {
    writeRequests.add(writeRequest);
    if (writeRequests.size() >= BATCH_SIZE) {
      flushBatch();
    }
  }

  // write lift ride events into DynamoDB
//  public void writeLiftRide(Map<String, AttributeValue> item) {
//    PutItemRequest putItemRequest = PutItemRequest.builder()
//        .tableName(TABLE_NAME)
//        .item(item)
//        .build();
//    try {
//      dynamoDbClient.putItem(putItemRequest);
//    } catch (DynamoDbException e) {
//      e.printStackTrace();
//      System.err.println(e.getMessage());
//    }
//  }

  // flush buffered write requests to DynamoDB
  private void flushBatch() {
    try {
      List<WriteRequest> batch = new ArrayList<>(writeRequests);
      writeRequests.clear();  // clear first to reduce lock time

      BatchWriteItemRequest batchWriteItemRequest = BatchWriteItemRequest.builder()
          .requestItems(Map.of(TABLE_NAME, batch))
          .build();
      dynamoDbClient.batchWriteItem(batchWriteItemRequest);
    } catch (DynamoDbException e) {
      System.err.println("Batch write failed: " + e.getMessage());
    }
  }

  //  Flush remaining requests (for shutdown)
  public synchronized void flushRemainingWrites() {
    if (!writeRequests.isEmpty()) {
      flushBatch();
    }
  }





}

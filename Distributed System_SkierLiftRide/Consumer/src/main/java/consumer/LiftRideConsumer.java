package consumer;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.rabbitmq.client.*;
import commons.shared.model.LiftRideMessage;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import software.amazon.awssdk.services.dynamodb.model.*;


public class LiftRideConsumer extends Thread {

  // Shared channel pool to allow multiple consumers to reuse RMQ channels
  private final BlockingQueue<Channel> channelPool;
  private final ObjectMapper objectMapper;
  private static final String QUEUE_NAME = "skier_queue";
  private static final SkierLiftRidesDao skierLiftRidesDao = SkierLiftRidesDao.getInstance();
  private static final LiftRideCountsDao liftRideCountsDao = LiftRideCountsDao.getInstance();

  public LiftRideConsumer(BlockingQueue<Channel> channelPool, ObjectMapper objectMapper) {
    this.channelPool = channelPool;
    this.objectMapper = objectMapper;
  }

  @Override
  public void run() {
    try {
      // Take a channel from the pool
      Channel channel = channelPool.take();
      // Define the message handler (callback) for incoming messages
      DeliverCallback deliverCallback = (consumerTag, delivery) -> {
        try {
          String message = new String(delivery.getBody(), "UTF-8");
          System.out.println("Received message: " + message);
          LiftRideMessage liftRideMessage = objectMapper.readValue(message, LiftRideMessage.class);
          Map<String, AttributeValue> item = convertLiftRideToAttributeValueMap(liftRideMessage);
//          skierLiftRidesDao.writeLiftRide(item);

          // Create a WriteRequest to insert into DynamoDB via BatchWriteItem
          WriteRequest writeRequest = WriteRequest.builder()
              .putRequest(PutRequest.builder().item(item).build())
              .build();

          // Add to batch (thread-safe)
          skierLiftRidesDao.addWriteRequest(writeRequest);

          // Atomic increment to LiftRideCounts (must be thread-safe)
          String resortSeasonDayId = liftRideMessage.getResortId() + "#" +
              liftRideMessage.getSeasonId() + "#" +
              liftRideMessage.getDayId();
          liftRideCountsDao.incrementBuffered(resortSeasonDayId);

          channel.basicAck(delivery.getEnvelope().getDeliveryTag(), false);
        } catch (Exception e) {
          // Log error and requeue the message
          System.err.println("Error processing message: " + e.getMessage());
          channel.basicNack(delivery.getEnvelope().getDeliveryTag(), false, true);
        }
      };

      channel.basicConsume(QUEUE_NAME, false, deliverCallback, consumerTag -> {
      });

      System.out.println("Consumer thread started: " + Thread.currentThread().getName());

      // Keep thread alive to listen for messages
      Thread.sleep(Long.MAX_VALUE);

    } catch (Exception e) {
      System.err.println("Consumer thread error: " + e.getMessage());
    } finally {
      // flush when consumer exits
      skierLiftRidesDao.flushRemainingWrites();
    }
  }

  /**
   * Convert LiftRideMessage to a DynamoDB item map.
   */
  private Map<String, AttributeValue> convertLiftRideToAttributeValueMap(
      LiftRideMessage liftRideMessage) {
    Map<String, AttributeValue> item = new HashMap<>();
    item.put("SkierID",
        AttributeValue.builder().n(String.valueOf(liftRideMessage.getSkierId())).build());
    item.put("SeasonDayTimeID",
        AttributeValue.builder().s(liftRideMessage.getSeasonId() + "#" + liftRideMessage.getDayId() + "#" + liftRideMessage.getLiftRide().getTime())
            .build());
    item.put("ResortID",
        AttributeValue.builder().n(String.valueOf(liftRideMessage.getResortId())).build());
    item.put("SeasonID",
        AttributeValue.builder().n(String.valueOf(liftRideMessage.getSeasonId())).build());
    item.put("LiftID",
        AttributeValue.builder().n(String.valueOf(liftRideMessage.getLiftRide().getLiftID()))
            .build());
    item.put("Time",
        AttributeValue.builder().n(String.valueOf(liftRideMessage.getLiftRide().getTime()))
            .build());

    item.put("ResortSeasonDayID",
        AttributeValue.builder()
            .s(liftRideMessage.getResortId() + "#" + liftRideMessage.getSeasonId() + "#"
                + liftRideMessage.getDayId()).build());
    return item;
  }
}

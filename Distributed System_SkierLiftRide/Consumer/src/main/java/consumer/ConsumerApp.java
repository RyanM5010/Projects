package consumer;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.rabbitmq.client.Channel;
import com.rabbitmq.client.Connection;
import com.rabbitmq.client.ConnectionFactory;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ArrayBlockingQueue;


public class ConsumerApp {
  private static final int NUM_THREADS = 64;
  private static final int POOL_SIZE = 64;
  private static final String QUEUE_NAME = "skier_queue";

  public static void main(String[] args) {
    ObjectMapper objectMapper = new ObjectMapper();

    ConnectionFactory factory = new ConnectionFactory();
    factory.setHost("172.31.30.196");
    factory.setUsername("sylvie");
    factory.setPassword("sylvieadmin");
//    factory.setHost("localhost");
//    factory.setUsername("guest");
//    factory.setPassword("guest");
    factory.setPort(5672);

    BlockingQueue<Channel> channelPool = new ArrayBlockingQueue<>(POOL_SIZE);
    try {
      Connection connection = factory.newConnection();
      for (int i = 0; i < POOL_SIZE; i++) {
        Channel channel = connection.createChannel();
        channel.queueDeclare(QUEUE_NAME, true, false, false, null);
        channel.basicQos(50);
        channelPool.add(channel);
      }
    } catch (Exception e) {
      System.err.println("Connection failed: " + e.getMessage());
    }

    // JVM shutdown hook
    Runtime.getRuntime().addShutdownHook(new Thread(() -> {
      System.out.println("Shutdown detected, flushing remaining writes...");
      SkierLiftRidesDao.getInstance().flushRemainingWrites();
    }));

    // Start consumer threads
    for (int i = 0; i < NUM_THREADS; i++) {
      new LiftRideConsumer(channelPool, objectMapper).start();
    }

    System.out.println("Consumers started with " + NUM_THREADS + " threads.");
  }


}

package skiers.service.rabbitmq;

import com.rabbitmq.client.AMQP;
import com.rabbitmq.client.Channel;
import com.rabbitmq.client.Connection;
import com.rabbitmq.client.ConnectionFactory;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import com.fasterxml.jackson.databind.ObjectMapper;
import commons.shared.model.LiftRide;
import commons.shared.model.LiftRideMessage;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;

@Service
public class RabbitMQService {
  private Connection connection;
  private static BlockingQueue<Channel> channelPool;
  private final ObjectMapper objectMapper;
  private static final String QUEUE_NAME = "skier_queue";
  private static final int POOL_SIZE = 32;

  @Value("${rabbitmq.host}")
  private String host;

  @Value("${rabbitmq.username}")
  private String username;

  @Value("${rabbitmq.password}")
  private String password;

  public RabbitMQService(ObjectMapper objectMapper) {
    this.objectMapper = objectMapper;
  }

  @PostConstruct
  public void init() throws Exception {
    // Initialize connection
    ConnectionFactory factory = new ConnectionFactory();
    factory.setHost(host);
    factory.setUsername(username);
    factory.setPassword(password);
    
    // Add these performance tuning parameters
    factory.setRequestedHeartbeat(30);
    factory.setConnectionTimeout(5000);
    factory.setNetworkRecoveryInterval(1000);
    factory.setAutomaticRecoveryEnabled(true);
    
    connection = factory.newConnection();
    
    // Declare queue once at startup
    try (Channel channel = connection.createChannel()) {
      channel.queueDeclare(QUEUE_NAME, true, false, false, null);
    }

    channelPool = new ArrayBlockingQueue<>(POOL_SIZE);
    // Pre-initialize pool
    for (int i = 0; i < POOL_SIZE; i++) {
      Channel channel = connection.createChannel();
      // No need to declare queue for each channel
      channelPool.add(channel);
    }
  }

  public void sendLiftRideMessage(int resortId, int seasonId, int dayId, int skierId, LiftRide liftRide) throws Exception {
    Channel channel = null;
    try {
      channel = channelPool.take();

      // Format data
      LiftRideMessage message = new LiftRideMessage(resortId, seasonId, dayId, skierId, liftRide);
      byte[] messageBytes = objectMapper.writeValueAsBytes(message);

      // Send to queue with non-persistent delivery mode (faster)
      channel.basicPublish("", QUEUE_NAME, 
          new AMQP.BasicProperties.Builder()
              .deliveryMode(1) // 1 = non-persistent (faster)
              .build(), 
          messageBytes);
    } catch (Exception e) {
      System.out.println(e.getMessage());
      throw new RuntimeException("Failed to publish message", e);
    } finally {
      if (channel != null) {
        try {
          // Non-blocking return to pool
          channelPool.offer(channel);
        } catch (Exception e) {
          throw new RuntimeException(e);
        }
      }
    }
  }

  @PreDestroy
  public void cleanup() throws Exception {
    try {
      for (Channel ch : channelPool) {
        if (ch != null) {
          ch.close();
        }
      }
      connection.close();
    } catch (Exception e) {
      System.err.println("Failed to close channels or connection: " + e.getMessage());
    }
  }
}

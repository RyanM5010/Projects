package skiers.service.rabbitmq;  // Updated package

import com.rabbitmq.client.Channel;
import com.rabbitmq.client.Connection;
import org.apache.commons.pool2.BasePooledObjectFactory;
import org.apache.commons.pool2.PooledObject;
import org.apache.commons.pool2.impl.DefaultPooledObject;

public class RabbitMQChannelFactory extends BasePooledObjectFactory<Channel> {
  private final Connection connection;

  public RabbitMQChannelFactory(Connection connection) {
    this.connection = connection;
  }

  @Override
  public Channel create() throws Exception {
    return connection.createChannel();
  }

  @Override
  public PooledObject<Channel> wrap(Channel channel) {
    return new DefaultPooledObject<>(channel);
  }

  @Override
  public boolean validateObject(PooledObject<Channel> pooledObject) {
    Channel channel = pooledObject.getObject();
    try {
      channel.basicQos(1); // Quick check to validate channel is responsive
      return channel.isOpen();
    } catch (Exception e) {
      return false; // Channel is broken, should be discarded
    }
  }

  @Override
  public void destroyObject(PooledObject<Channel> pooledObject) throws Exception {
    Channel channel = pooledObject.getObject();
    if (channel.isOpen()) {
      channel.close();
    }
  }
} 
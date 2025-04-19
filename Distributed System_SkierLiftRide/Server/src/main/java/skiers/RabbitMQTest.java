package skiers;

import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.WebApplicationType;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.context.ConfigurableApplicationContext;
import commons.shared.model.LiftRide;
import skiers.service.rabbitmq.RabbitMQService;

@SpringBootApplication
public class RabbitMQTest {

    public static void main(String[] args) {
        // Start Spring context without web server
        ConfigurableApplicationContext context = new SpringApplicationBuilder(RabbitMQTest.class)
            .web(WebApplicationType.NONE)
            .run(args);
        
        // Get RabbitMQ service
        RabbitMQService rabbitMQService = context.getBean(RabbitMQService.class);
        
        try {
            // Create test lift ride
            LiftRide liftRide = new LiftRide(21, 217);

            System.out.println("Starting to send test messages...");
            
            // Send 5 test messages
            for (int i = 0; i < 5; i++) {
                rabbitMQService.sendLiftRideMessage(1, 2022, 1, 123, liftRide);
                System.out.println("Sent message " + i);
            }
            
            System.out.println("Test completed successfully!");
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        } finally {
            // Clean shutdown
            context.close();
        }
    }
}
